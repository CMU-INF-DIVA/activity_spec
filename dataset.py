import json
import random
from collections import namedtuple

import numpy as np
import torch
from avi_r import AVIReader
from torch.utils.data import Dataset
from pyturbo import progressbar

from .base import ActivityTypes, ProposalType
from .cube import CubeActivities


class VideoDataset(Dataset):

    def __init__(self, file_index_path, proposal_dir, label_dir=None,
                 dataset='MEVA'):
        self.file_index_path = file_index_path
        with open(file_index_path) as f:
            self.file_index = [*json.load(f).items()]
        self.proposal_dir = proposal_dir
        self.label_dir = label_dir
        self.dataset = dataset
        self.activity_types = ActivityTypes[dataset]

    def __getitem__(self, idx):
        video_name, video_meta = self.file_index[idx]
        proposals = CubeActivities.load(
            video_name, self.proposal_dir, ProposalType)
        if self.label_dir is not None:
            labels = CubeActivities.load(
                video_name, self.label_dir, None)
        else:
            labels = None
        return video_name, video_meta, proposals, labels

    def __len__(self):
        return len(self.file_index)


Proposal = namedtuple('Proposal', ['video_name', 'localization', 'label'])


class ProposalDataset(Dataset):

    def __init__(self, file_index_path, proposal_dir, label_dir, video_dir,
                 dataset='MEVA', *, negative_fraction=None, enlarge_rate=None,
                 stride=1, clip_transform=None, label_transform=None):
        self.video_dataset = VideoDataset(
            file_index_path, proposal_dir, label_dir, dataset)
        self.video_dir = video_dir
        self.negative_fraction = negative_fraction
        self.enlarge_rate = enlarge_rate
        self.stride = stride
        self.clip_transform = clip_transform
        self.label_transform = label_transform
        self.load_proposals()

    def load_proposals(self):
        self.positive_proposals = []
        self.negative_proposals = []
        for video_name, _, proposals, labels in progressbar(
                self.video_dataset, 'Load proposals'):
            columns = proposals.columns
            localizations = proposals.cubes[:, columns.t0:columns.y1 + 1]
            for localization, label in zip(localizations, labels.cubes):
                proposal = Proposal(video_name, localization, label)
                if label[0] > 0:
                    self.negative_proposals.append(proposal)
                else:
                    self.positive_proposals.append(proposal)
        if self.negative_fraction is None:
            self.negative_quota = len(self.negative_proposals)
            indices = np.concatenate([
                np.arange(len(self.positive_proposals)),
                np.full(self.negative_quota, -1, dtype=np.int)])
            self.proposal_indices = np.random.permutation(indices)
        else:
            self.negative_quota = int(round(
                len(self.positive_proposals) * self.negative_fraction))
            self.proposal_indices = np.concatenate([
                np.arange(len(self.positive_proposals)),
                np.arange(-len(self.negative_proposals), 0)])

    def load_frames(self, video_name, t0, t1):
        '''
        Return frames as T x H x W x C[RGB] in [0, 256)
        '''
        video = AVIReader(video_name, self.video_dir)
        frames = []
        num_frames = (t1 - t0) // self.stride
        video.seek(t0)
        for frame in video.get_iter(num_frames, self.stride):
            frames.append(frame.numpy('rgb24'))
        if len(frames) < num_frames:
            frames.extend(frames[-1:] * (num_frames - len(frames)))
        frames = np.stack(frames)
        return frames

    def __getitem__(self, idx):
        proposal_index = self.proposal_indices[0]
        if proposal_index >= 0:
            proposal = self.positive_proposals[proposal_index]
        elif self.negative_fraction is None:
            proposal = self.negative_proposals[proposal_index]
        else:
            proposal = random.choice(self.negative_proposals)
        t0, t1, x0, y0, x1, y1 = proposal.localization.tolist()
        frames = self.load_frames(proposal.video_name, int(t0), int(t1))
        clip = torch.as_tensor(frames[
            :, int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))])
        label = proposal.label
        if self.clip_transform is not None:
            clip = self.clip_transform(clip)
        if self.label_transform is not None:
            label = self.label_transform(label)
        return clip, label

    def __len__(self):
        return len(self.positive_proposals) + self.negative_quota
