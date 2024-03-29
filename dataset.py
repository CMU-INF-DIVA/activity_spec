import json
import os.path as osp
import random
import warnings
from collections import namedtuple

import decord
import numpy as np
import psutil
import torch
from avi_r import AVIReader
from decord import VideoReader, cpu, gpu
from torch.utils.data import Dataset

from .base import ActivityTypes, ProposalType
from .cube import CubeActivities, CubeColumns


class VideoDataset(Dataset):

    def __init__(self, file_index_path, proposal_dir, label_dir=None,
                 dataset='MEVA', proposal_columns=CubeColumns):
        self.file_index_path = file_index_path
        with open(file_index_path) as f:
            self.file_index = [*json.load(f).items()]
        self.proposal_dir = proposal_dir
        self.label_dir = label_dir
        self.dataset = dataset
        self.proposal_columns = proposal_columns
        self.activity_types = ActivityTypes[dataset]

    def __getitem__(self, idx):
        video_name, video_meta = self.file_index[idx]
        proposals = CubeActivities.load(
            video_name, self.proposal_dir, ProposalType, self.proposal_columns)
        if self.label_dir is not None:
            labels = CubeActivities.load(
                video_name, self.label_dir, None, self.activity_types)
        else:
            labels = CubeActivities(
                torch.full((len(proposals), len(self.activity_types)), -1.),
                video_name, None, self.activity_types)
        return video_name, video_meta, proposals, labels

    def __len__(self):
        return len(self.file_index)


Sample = namedtuple('Sample', [
    'video_name', 'index', 'proposal', 'label'])


class ProposalDataset(Dataset):

    def __init__(self, file_index_path, proposal_dir, label_dir, video_dir,
                 clips_dir=None, dataset='MEVA', *, eval_mode=False,
                 negative_fraction=None, negative_whitelist=None,
                 spatial_enlarge_rate=None, frame_stride=1, clip_duration=None,
                 clip_transform=None, label_transform=None, device=None):
        assert label_dir is not None or eval_mode
        self.video_dataset = VideoDataset(
            file_index_path, proposal_dir, label_dir, dataset)
        self.video_dir = video_dir
        self.clips_dir = clips_dir
        self.dataset = dataset
        self.eval_mode = eval_mode
        self.negative_fraction = negative_fraction
        self.negative_whitelist = negative_whitelist
        self.spatial_enlarge_rate = spatial_enlarge_rate
        self.frame_stride = frame_stride
        self.clip_duration = clip_duration
        self.clip_transform = clip_transform
        self.label_transform = label_transform
        self.load_samples()
        self.device = device
        self.cache = (None, None)

    def load_samples(self):
        self.proposals = []
        self.all_samples = []
        self.positive_samples = []
        self.negative_samples = []
        self.num_frames = 0
        for video_name, video_meta, proposals, labels in self.video_dataset:
            start_end = {
                v: int(k) - 1 for k, v in video_meta['selected'].items()}
            self.num_frames += start_end[0] - start_end[1]
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
                    if self.negative_whitelist is None or \
                            video_name in self.negative_whitelist:
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
            frame_stride = self.frame_stride
            while frame_stride >= 1:
                clip_name = '%s.%d-%d_%d.mp4' % (
                    osp.splitext(video_name)[0], t0, t1, frame_stride)
                clip_path = osp.join(self.clips_dir, video_name, clip_name)
                if osp.exists(clip_path):
                    decord_context = cpu(0)
                    if self.device is not None and self.device.type == 'cuda':
                        decord_context = gpu(self.device.index)
                    with decord.bridge.use_torch():
                        video = VideoReader(clip_path, ctx=decord_context)
                        frames = video.get_batch(range(len(video)))
                        del video
                    frames = frames[::self.frame_stride // frame_stride]
                    return frames
                frame_stride //= 2
            else:
                warnings.warn(
                    f'Clip({video_name}, {t0}-{t1}) not found in '
                    f'clips_dir: {self.clips_dir}')
        if self.dataset == 'MEVA':
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
        else:
            with decord.bridge.use_torch():
                video = VideoReader(osp.join(self.video_dir, video_name))
                frame_ids = np.arange(t0, t1, self.frame_stride)
                frames = video.get_batch(frame_ids)
                del video
        return frames

    def load_frames_cache(self, video_name, t0, t1):
        if self.cache[0] == (video_name, t0, t1):
            return self.cache[1]
        frames = self.load_frames(video_name, t0, t1)
        self.cache = ((video_name, t0, t1), frames)
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
        if self.clip_duration is not None:
            t1 = max(t1, t0 + self.clip_duration)  # pad t1
        frames = self.load_frames_cache(sample.video_name, int(t0), int(t1))
        clip_ = frames[:, int(y0):int(np.ceil(y1)), int(x0):int(np.ceil(x1))]
        clip = clip_.cpu()
        del frames, clip_
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
