import json
import os.path as osp
from collections import namedtuple

import numpy as np
import torch
from avi_r import AVIReader
from torch.utils.data import Dataset, IterableDataset, Subset, get_worker_info

from .base import ActivityTypes, ProposalType
from .cube import CubeActivities


class VideoDataset(Dataset):

    def __init__(self, file_index_path, video_dir, proposal_dir, label_dir,
                 dataset='MEVA'):
        with open(file_index_path) as f:
            self.file_index = [*json.load(f).items()]
        self.video_dir = video_dir
        self.proposal_dir = proposal_dir
        self.label_dir = label_dir
        self.dataset = dataset
        self.activity_types = ActivityTypes[dataset]

    def __getitem__(self, idx):
        video_name, _ = self.file_index[idx]
        proposal = CubeActivities.load(
            video_name, self.proposal_dir, ProposalType)
        label = CubeActivities.load(
            video_name, self.label_dir, None)
        video_path = osp.join(self.video_dir, video_name)
        return video_path, proposal, label


Frame = namedtuple('Frame', ['id', 'image'])


class ProposalInVideo(IterableDataset):

    def __init__(self, video_path, proposal, label,
                 enlarge_rate=None, stride=1):
        self.video_path = video_path
        self.proposal = proposal
        self.label = label
        self.enlarge_rate = enlarge_rate
        self.stride = stride

    def load_frames(self, video, t0, t1, frame_buffer):
        '''
        Return frames as T x H x W x C[RGB] in [0, 256)
        '''
        frame_buffer = [*filter(lambda f: f.id >= t0, frame_buffer)]
        num_frames = (t1 - t0) // self.stride
        if len(frame_buffer) < num_frames:
            if len(frame_buffer) == 0:
                video.seek(t0)
            if isinstance(video, AVIReader):
                for frame in video.get_iter(
                        num_frames - len(frame_buffer), self.stride):
                    frame = Frame(frame.frame_id, frame.numpy('rgb24'))
                    frame_buffer.append(frame)
            else:
                raise NotImplementedError(type(video))
        if len(frame_buffer) < num_frames:
            frame_buffer.extend(frame_buffer[-1:] * (
                num_frames - len(frame_buffer)))
        return np.stack([f.image for f in frame_buffer])

    def __iter__(self):
        video = AVIReader(self.video_path)
        frame_buffer = []
        if self.enlarge_rate is not None:
            proposal = self.proposal.spatial_enlarge(
                self.enlarge_rate, (video.width, video.height))
        else:
            proposal = self.proposal
        columns = proposal.columns
        for prop, label in zip(proposal.cubes, self.label.cubes):
            t0, t1 = prop[columns.t0:columns.t1 + 1].type(torch.int).tolist()
            x0, y0 = prop[columns.x0:columns.y0 + 1].type(torch.int).tolist()
            x1, y1 = prop[columns.x1:columns.y1 + 1].ceil().type(
                torch.int).tolist()
            frames = self.load_frames(video, t0, t1, frame_buffer)
            clip = torch.as_tensor(frames[:, y0:y1, x0:x1])
            yield clip, label
        video.close()
        frame_buffer.clear()

    def __len__(self):
        return len(self.proposal)


class ProposalDataset(IterableDataset):

    def __init__(self, file_index_path, video_dir, proposal_dir, label_dir,
                 dataset='MEVA', enlarge_rate=0.1, stride=2,
                 clip_transform=None, label_transform=None):
        self.video_dataset = VideoDataset(
            file_index_path, video_dir, proposal_dir, label_dir, dataset)
        self.enlarge_rate = enlarge_rate
        self.stride = stride
        self.clip_transform = clip_transform
        self.label_transform = label_transform

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is not None:
            indices = [*range(worker_info.id, len(self.video_dataset),
                              worker_info.num_workers)]
            video_dataset = Subset(self.video_dataset, indices)
        else:
            video_dataset = self.video_dataset
        for video_path, proposal, label in video_dataset:
            for clip, label in ProposalInVideo(video_path, proposal, label,
                                               self.enlarge_rate, self.stride):
                if self.clip_transform is not None:
                    clip = self.clip_transform(clip)
                if self.label_transform is not None:
                    label = self.label_transform(label)
                yield clip, label
