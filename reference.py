import gzip
import json

import numpy as np
import torch

from .base import ActivityType
from .cube import CubeActivities


class Reference(object):

    def __init__(self, reference_path: str, frame_rate: int = 30):
        self.frame_rate = frame_rate
        with gzip.open(reference_path, 'rt', encoding='utf-8') as f:
            reference = json.load(f)
        self.video_list = reference['filesProcessed']
        self.activities = self._split_by_video(reference['activities'])

    def _split_by_video(self, activities):
        activities_by_video = {v: [] for v in self.video_list}
        for activity in activities:
            video_name = [*activity['localization'].keys()][0]
            activities_by_video[video_name].append(activity)
        return activities_by_video

    def get_quantized_cubes(self, video_name: str, stride: int = 32,
                            box_mode: str = 'start'):
        raw_activities = self.activities[video_name]
        if len(raw_activities) == 0:
            return CubeActivities(torch.empty((0, 7), dtype=torch.int),
                                  video_name, ActivityType)
        quantized_activities = []
        for activity in raw_activities:
            activity_type = ActivityType[activity['activity']]
            start_end = {v: int(k) for k, v in activity['localization'][
                video_name].items()}
            start, end = start_end[1], start_end[0]
            length = end - start
            cube_starts = np.arange(
                start // stride, end // stride + 1) * stride
            cube_ends = cube_starts + stride
            activity_starts = cube_starts.copy()
            activity_starts[0] = start
            activity_ends = cube_ends.copy()
            activity_ends[-1] = end
            if length < self.frame_rate:
                # intersection >= 50% * reference
                valid = (activity_ends - activity_starts) >= length / 2
            else:
                # intersection >= 1 second
                valid = (activity_ends - activity_starts) >= self.frame_rate
            for cube_i in range(valid.shape[0]):
                if valid[cube_i]:
                    quantized_activity = np.empty(7, dtype=np.int)
                    quantized_activity[0] = activity_type
                    quantized_activity[1:5] = self._get_box(
                        activity, video_name, activity_starts[cube_i],
                        activity_ends[cube_i], box_mode)
                    quantized_activity[5] = cube_starts[cube_i]
                    quantized_activity[6] = cube_ends[cube_i]
                    quantized_activities.append(quantized_activity)
        quantized_activities = torch.as_tensor(
            np.stack(quantized_activities), dtype=torch.int)
        quantized_cubes = CubeActivities(
            quantized_activities, video_name, ActivityType)
        return quantized_cubes

    def _get_box(self, activity, video_name, start, end, mode):
        if mode == 'start':
            return self._get_box_at(activity, video_name, start)
        else:
            raise NotImplementedError(mode)

    def _get_box_at(self, activity, video_name, frame_id):
        frame_id = str(frame_id)
        boxes = []
        for obj in activity['objects']:
            obj = obj['localization'][video_name]
            if frame_id not in obj or 'boundingBox' not in obj[frame_id]:
                continue
            box = obj[frame_id]['boundingBox']
            box = np.array([box['x'], box['y'], box['w'], box['h']],
                           dtype=np.int)
            box[2:] += box[:2]
            boxes.append(box)
        if len(boxes) == 1:
            return boxes[0]
        boxes = np.stack(boxes)
        box = np.empty(4)
        box[:2] = boxes[:, :2].min()
        box[2:] = boxes[:, 2:].max()
        return box
