import gzip
import json
from enum import Enum, EnumMeta

import numpy as np
import torch

from .cube import CubeActivities


class Reference(object):

    def __init__(self, reference_path: str, type_names: EnumMeta,
                 frame_rate: int = 30):
        self.type_names = type_names
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

    def get_quantized_cubes(self, video_name: str, cube_length: int):
        '''
        Convert reference into quantized cubes with fixed length.
        Cube score is the temporal overlap between a cube and the reference.
        Spatial size is the union of all frames in the clip.
        A cube is ignored if it can never be matched by the scorer.
        '''
        raw_activities = self.activities[video_name]
        if len(raw_activities) == 0:
            return CubeActivities(torch.empty((0, 8)), video_name,
                                  self.type_names)
        quantized_activities = []
        for activity in raw_activities:
            activity_type = self.type_names[activity['activity']]
            start_end = {v: int(k) for k, v in activity['localization'][
                video_name].items()}
            start, end = start_end[1], start_end[0]
            length = end - start
            cube_starts = np.arange(
                start // cube_length, end // cube_length + 1) * cube_length
            cube_ends = cube_starts + cube_length
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
                    box = self._get_box(
                        activity, video_name,
                        activity_starts[cube_i], activity_ends[cube_i])
                    if box is None:
                        continue
                    overlap = (activity_ends[cube_i] -
                               activity_starts[cube_i]) / cube_length
                    quantized_activity = np.empty(8, dtype=np.float32)
                    quantized_activity[0] = activity_type
                    quantized_activity[1] = overlap
                    quantized_activity[2] = cube_starts[cube_i]
                    quantized_activity[3] = cube_ends[cube_i]
                    quantized_activity[4:8] = box
                    quantized_activities.append(quantized_activity)
        quantized_activities = torch.as_tensor(np.stack(quantized_activities))
        quantized_cubes = CubeActivities(
            quantized_activities, video_name, self.type_names)
        return quantized_cubes

    def _get_box(self, activity, video_name, start, end):
        boxes = []
        for frame_id in range(start, end):
            frame_id = str(frame_id)
            for obj in activity['objects']:
                obj = obj['localization'][video_name]
                if frame_id not in obj or 'boundingBox' not in obj[frame_id]:
                    continue
                box = obj[frame_id]['boundingBox']
                box = np.array([box['x'], box['y'], box['w'], box['h']],
                               dtype=np.int)
                box[2:] += box[:2]
                boxes.append(box)
        if len(boxes) == 0:
            return None
        boxes = np.stack(boxes)
        box = np.empty(4)
        box[:2] = boxes[:, :2].min(axis=0)
        box[2:] = boxes[:, 2:].max(axis=0)
        return box
