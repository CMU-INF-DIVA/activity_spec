import gzip
import json
from enum import EnumMeta
from typing import Union

import numpy as np
import torch

from .cube import CubeActivities, CubeColumns


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

    def get_quantized_cubes(self, video_name: str, cube_length: int,
                            stride: Union[None, int] = None):
        '''
        Convert reference into quantized cubes with fixed length and stride.
        By default stride equals to cube length, so cubes are non-overlapped.
        Cube score is the temporal overlap between a cube and the reference.
        Spatial size is the union of all frames in the clip.
        '''
        stride = stride or cube_length
        raw_activities = self.activities[video_name]
        if len(raw_activities) == 0:
            return CubeActivities(
                torch.empty((0, len(CubeColumns))), video_name, self.type_names)
        quantized_activities = []
        for act_id, activity in enumerate(raw_activities):
            activity_type = self.type_names[activity['activity']]
            start_end = {v: int(k) for k, v in activity['localization'][
                video_name].items()}
            activity_start, activity_end = start_end[1], start_end[0]
            first_cube_idx = int(max(0, np.ceil(
                (activity_start - cube_length + 1) / stride)))
            last_cube_idx = int(np.floor((activity_end - 1) / stride))
            cube_starts = np.arange(
                first_cube_idx, last_cube_idx + 1) * stride
            cube_ends = cube_starts + cube_length
            activity_starts = np.maximum(cube_starts, activity_start)
            activity_ends = np.minimum(cube_ends, activity_end)
            for cube_i in range(cube_starts.shape[0]):
                box = self._get_box(
                    activity, video_name,
                    activity_starts[cube_i], activity_ends[cube_i])
                if box is None:
                    continue
                overlap = (activity_ends[cube_i] -
                           activity_starts[cube_i]) / cube_length
                quantized_activity = np.empty(
                    len(CubeColumns), dtype=np.float32)
                quantized_activity[CubeColumns.id] = act_id
                quantized_activity[CubeColumns.type] = activity_type
                quantized_activity[CubeColumns.score] = overlap
                quantized_activity[CubeColumns.t0] = cube_starts[cube_i]
                quantized_activity[CubeColumns.t1] = cube_ends[cube_i]
                quantized_activity[CubeColumns.x0:CubeColumns.y1 + 1] = box
                quantized_activities.append(quantized_activity)
        quantized_activities = torch.as_tensor(np.stack(quantized_activities))
        quantized_cubes = CubeActivities(
            quantized_activities, video_name, self.type_names, CubeColumns)
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
