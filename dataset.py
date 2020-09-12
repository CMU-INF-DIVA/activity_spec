import json
import os.path as osp
import random
import warnings
from collections import namedtuple

import decord
import numpy as np
import torch
from decord import VideoReader
from avi_r import AVIReader
from torch.utils.data import Dataset

from .base import ActivityTypes, ProposalType
from .cube import CubeActivities, CubeColumns

decord.bridge.set_bridge('torch')


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


Sample = namedtuple('Sample', [
    'video_name', 'index', 'proposal', 'label'])


class ProposalDataset(Dataset):

    def __init__(self, file_index_path, proposal_dir, label_dir, video_dir,
                 clips_dir=None, dataset='MEVA', *, eval_mode=False,
                 negative_fraction=None, spatial_enlarge_rate=None,
                 frame_stride=1, clip_transform=None, label_transform=None):
        self.video_dataset = VideoDataset(
            file_index_path, proposal_dir, label_dir, dataset)
        self.video_dir = video_dir
        self.clips_dir = clips_dir
        self.eval_mode = eval_mode
        self.negative_fraction = negative_fraction
        self.spatial_enlarge_rate = spatial_enlarge_rate
        self.frame_stride = frame_stride
        self.clip_transform = clip_transform
        self.label_transform = label_transform
        self.load_samples()

    def load_samples(self):
        self.proposals = []
        self.all_samples = []
        self.positive_samples = []
        self.negative_samples = []
        for video_name, _, proposals, labels in self.video_dataset:
            self.proposals.append(proposals)
            if self.spatial_enlarge_rate is not None:
                proposals = proposals.spatial_enlarge(
                    self.spatial_enlarge_rate)
            columns = [proposals.columns[c.name] for c in CubeColumns]
            for i, (proposal, label) in enumerate(
                    zip(proposals.cubes[:, columns], labels.cubes)):
                sample = Sample(video_name, i, proposal, label)
                self.all_samples.append(sample)
                if self.eval_mode:  # Use all samples in eval mode
                    continue
                if label[0] > 0:
                    self.negative_samples.append(sample)
                elif label[1:].sum() > 0:
                    self.positive_samples.append(sample)
        if self.eval_mode:
            return
        negative_quota = len(self.negative_samples)
        if self.negative_fraction is not None:
            negative_quota = int(round(
                len(self.positive_samples) * self.negative_fraction))
        # idx > 0 --> positive_samples[idx - 1]
        # idx == 0 --> Random sample from negative_samples
        # idx < 0 --> negative_samples[idx]
        if negative_quota < len(self.negative_samples):
            indices = np.concatenate([
                np.arange(1, len(self.positive_samples) + 1),
                np.zeros(negative_quota, dtype=np.int)])
            self.sample_indices = np.random.permutation(indices)
        else:
            self.sample_indices = np.concatenate([
                np.arange(1, len(self.positive_samples) + 1),
                np.arange(-len(self.negative_samples), 0)])

    def load_frames(self, video_name, t0, t1):
        '''
        Return frames as T x H x W x C[RGB] in [0, 256)
        '''
        if self.clips_dir is not None:
            clip_name = '%s.%d-%d_%d.mp4' % (
                osp.splitext(video_name)[0], t0, t1, self.frame_stride)
            clip_path = osp.join(self.clips_dir, video_name, clip_name)
            if osp.exists(clip_path):
                video = VideoReader(clip_path)
                frames = torch.stack([video[i] for i in range(len(video))])
                return frames
            else:
                warnings.warn(
                    'Clip not found in clips_dir: %s' % (self.clips_dir))
        video = AVIReader(video_name, self.video_dir)
        frames = []
        num_frames = (t1 - t0) // self.frame_stride
        video.seek(t0)
        for frame in video.get_iter(num_frames, self.frame_stride):
            frames.append(frame.numpy('rgb24'))
        if len(frames) < num_frames:
            frames.extend(frames[-1:] * (num_frames - len(frames)))
        frames = torch.as_tensor(np.stack(frames))
        video.close()
        return frames

    def __getitem__(self, idx):
        if self.eval_mode:
            sample = self.all_samples[idx]
        else:
            sample_index = self.sample_indices[idx]
            if sample_index > 0:
                sample = self.positive_samples[sample_index - 1]
            elif sample_index < 0:
                sample = self.negative_samples[sample_index]
            else:
                sample = random.choice(self.negative_samples)
        t0, t1, x0, y0, x1, y1 = sample.proposal[
            CubeColumns.t0:CubeColumns.y1 + 1].tolist()
        frames = self.load_frames(sample.video_name, int(t0), int(t1))
        clip = frames[:, int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))]
        label = sample.label
        if self.clip_transform is not None:
            clip = self.clip_transform(clip)
        if self.label_transform is not None:
            label = self.label_transform(label)
        return clip, label

    def __len__(self):
        if self.eval_mode:
            return len(self.all_samples)
        return self.sample_indices.shape[0]
