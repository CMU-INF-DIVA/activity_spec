import os.path as osp
from enum import EnumMeta, IntEnum, auto
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .base import PROP_TYPE_SCALE


class CubeColumns(IntEnum):

    id = 0
    type = auto()
    score = auto()
    t0 = auto()
    t1 = auto()
    x0 = auto()
    y0 = auto()
    x1 = auto()
    y1 = auto()


class CubeActivities(object):

    '''
    Activities each as a spatial-temporal cube.
    cubes: [(id, type, score, t0, t1, x0, y0, x1, y1)].
    Although stored as float, id and type should be int values.
    '''

    def __init__(self, cubes: torch.Tensor, video_name: str,
                 type_names: Optional[EnumMeta], columns: EnumMeta, *,
                 sub_cubes: Optional[List[torch.Tensor]] = None):
        assert cubes.ndim == 2 and cubes.shape[1] == len(columns), \
            'Cube activity format invalid'
        self.cubes = cubes
        self.video_name = video_name
        self.type_names = type_names
        self.columns = columns
        self.sub_cubes = sub_cubes

    def __len__(self):
        return self.cubes.shape[0]

    def __repr__(self):
        return '%s(%d@%s)' % (
            self.__class__.__name__, len(self), self.video_name)

    def to_internal(self):
        '''
        Internal storage format as pd.DataFrame.
        '''
        df = pd.DataFrame(self.cubes.cpu().numpy(), columns=[
            c.name for c in self.columns])
        if 'type' in df and self.type_names is not None:
            df['type'] = df['type'].apply(lambda v: self.type_names(v).name)
        return df

    def _get_official_boxes(self, cubes):
        int_cubes = torch.cat([
            cubes[:, [self.columns.t0]].type(torch.int) + 1,
            cubes[:, [self.columns.x0, self.columns.y0]].type(
                torch.int),
            cubes[:, [self.columns.x1, self.columns.y1]].ceil().type(
                torch.int)],
            axis=1)
        boxes = {}
        for cube in int_cubes:
            t0, x0, y0, x1, y1 = cube.tolist()
            boxes[str(t0)] = {'boundingBox': {
                'x': x0, 'y': y0, 'w': x1 - x0, 'h': y1 - y0}}
        return boxes

    def to_official(self, object_types=None):
        '''
        Official format in Json structure,
        only contains temporal and type information.
        Note: the frame id in the official format starts from 1.
        '''
        activities = []
        for cube_i, cube in enumerate(self.cubes):
            obj_id = cube[self.columns.id].item()
            activity_type = self.type_names(
                int(round(cube[self.columns.type].item()))).name
            if object_types is not None:
                object_type = object_types(
                    int(round(obj_id % 1 * PROP_TYPE_SCALE))).name
            else:
                object_type = 'Any'
            obj_id = int(round(obj_id))
            score = cube[self.columns.score].item()
            t0, t1 = (cube[[self.columns.t0, self.columns.t1]] + 1).type(
                torch.int).tolist()
            if self.sub_cubes is not None:
                boxes = self._get_official_boxes(self.sub_cubes[cube_i])
            else:
                boxes = self._get_official_boxes(cube[None])
            boxes[str(t1)] = {}
            activity = {
                'activity': activity_type, 'presenceConf': score,
                'localization': {self.video_name: {str(t0): 1, str(t1): 0}},
                'objects': [
                    {'objectType': object_type, 'objectID': obj_id,
                     'localization': {self.video_name: boxes}}]}
            activities.append(activity)
        return activities

    def save(self, save_dir: str, suffix: str = ''):
        '''
        Save as csv file in save_dir.
        '''
        # TODO: save sub_cubes if present
        df = self.to_internal()
        filename = self._get_internal_filename(
            self.video_name, suffix, save_dir)
        df.to_csv(filename)

    @classmethod
    def load(cls, video_name: str, load_dir: str,
             type_names: Optional[EnumMeta],
             columns: Optional[EnumMeta] = None,
             suffix: str = ''):
        '''
        Load from csv file in load_dir.
        '''
        filename = cls._get_internal_filename(video_name, suffix, load_dir)
        df = pd.read_csv(filename, index_col=0)
        if 'type' in df and type_names is not None:
            df['type'] = df['type'].apply(lambda v: type_names[v].value)
        cubes = torch.as_tensor(df.values.astype(np.float32))
        if columns is None:
            columns = IntEnum('loaded_columns', [
                (c, i) for i, c in enumerate(df.columns)])
        obj = cls(cubes, video_name, type_names, columns)
        return obj

    def spatial_enlarge(self, enlarge_rate: float,
                        spatial_limit: Optional[Tuple] = None):
        '''
        Enlarge spatial boxes.
        enlarge_rate: enlarge rate in each axis.
        spatial_limit: (x, y), frame size.
        '''
        spatial_size = self.cubes[:, [self.columns.x1, self.columns.y1]] - \
            self.cubes[:, [self.columns.x0, self.columns.y0]]
        enlarge_size = spatial_size * enlarge_rate
        new_cubes = self.cubes.clone()
        new_cubes[:, [self.columns.x0, self.columns.y0]] = torch.clamp(
            self.cubes[:, [self.columns.x0, self.columns.y0]] - enlarge_size,
            min=0)
        new_cubes[:, [self.columns.x1, self.columns.y1]] = \
            self.cubes[:, [self.columns.x1, self.columns.y1]] + enlarge_size
        if spatial_limit is not None:
            new_cubes[:, [self.columns.x1, self.columns.y1]] = torch.min(
                new_cubes[:, [self.columns.x1, self.columns.y1]],
                torch.as_tensor([spatial_limit], dtype=torch.float))
        return self.duplicate_with(new_cubes)

    def duplicate_with(self, cubes: Optional[torch.Tensor] = None, *,
                       selection: Optional[torch.Tensor] = None,
                       video_name: Optional[str] = None,
                       type_names: Optional[EnumMeta] = None,
                       columns: EnumMeta = None,
                       sub_cubes: Optional[List[torch.Tensor]] = None):
        '''
        Create a new instance with the same attributes unless specified.
        Cubes can be selected via the selection argument.
        '''
        cubes = cubes if cubes is not None else self.cubes
        if selection is not None:
            cubes = cubes[selection]
        video_name = video_name or self.video_name
        type_names = type_names or self.type_names
        columns = columns or self.columns
        sub_cubes = sub_cubes or self.sub_cubes
        if selection is not None and sub_cubes is not None:
            mask = torch.zeros(len(sub_cubes), dtype=torch.bool)
            mask[selection] = True
            sub_cubes = [sub_cubes[i] for i in mask.nonzero(as_tuple=True)[0]]
        return type(self)(cubes, video_name, type_names, columns,
                          sub_cubes=sub_cubes)

    def merge_with(self, cube_acts_list: List, **kwargs):
        cubes = torch.cat([self.cubes] + [c.cubes for c in cube_acts_list])
        return self.duplicate_with(cubes, **kwargs)

    @staticmethod
    def _get_internal_filename(video_name, suffix, data_dir):
        return osp.join(data_dir, osp.splitext(video_name)[0] + suffix + '.csv')
