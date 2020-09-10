import os.path as osp
from enum import EnumMeta, IntEnum, auto
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch


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
                 type_names: Union[None, EnumMeta], columns: EnumMeta):
        assert cubes.ndim == 2 and cubes.shape[1] == len(columns), \
            'Cube activity format invalid'
        self.cubes = cubes
        self.video_name = video_name
        self.type_names = type_names
        self.columns = columns

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

    def to_official(self):
        '''
        Official format in Json structure, 
        only contains temporal and type information.
        '''
        activities = []
        for cube in self.cubes:
            activity_type = self.type_names(
                int(round(cube[self.columns.type].item()))).name
            score = cube[self.columns.score].item()
            t0 = int(cube[self.columns.t0].item())
            t1 = int(cube[self.columns.t1].item())
            activity = {
                'activity': activity_type, 'presenceConf': score,
                'localization': {self.video_name: {str(t0): 1, str(t1): 0}}}
            activities.append(activity)
        return activities

    def save(self, save_dir: str, suffix: str = ''):
        '''
        Save as csv file in save_dir.
        '''
        df = self.to_internal()
        filename = self._get_internal_filename(
            self.video_name, suffix, save_dir)
        df.to_csv(filename)

    @classmethod
    def load(cls, video_name: str, load_dir: str,
             type_names: Union[None, EnumMeta], suffix: str = ''):
        '''
        Load from csv file in load_dir.
        '''
        filename = cls._get_internal_filename(video_name, suffix, load_dir)
        df = pd.read_csv(filename, index_col=0)
        if 'type' in df and type_names is not None:
            df['type'] = df['type'].apply(lambda v: type_names[v].value)
        cubes = torch.as_tensor(df.values.astype(np.float32))
        columns = IntEnum('loaded_columns', [
            (c, i) for i, c in enumerate(df.columns)])
        obj = cls(cubes, video_name, type_names, columns)
        return obj

    def spatial_enlarge(self, enlarge_rate: float,
                        spatial_limit: Union[None, Tuple] = None):
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

    def duplicate_with(self, cubes: Union[None, torch.Tensor] = None, *,
                       selection: Union[None, torch.Tensor] = None,
                       video_name: Union[None, str] = None,
                       type_names: Union[None, EnumMeta] = None,
                       columns: EnumMeta = None):
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
        return type(self)(cubes, video_name, type_names, columns)

    def merge_with(self, cube_acts_list: List, **kwargs):
        cubes = torch.cat([self.cubes] + [c.cubes for c in cube_acts_list])
        return self.duplicate_with(cubes, **kwargs)

    @staticmethod
    def _get_internal_filename(video_name, suffix, data_dir):
        return osp.join(data_dir, osp.splitext(video_name)[0] + suffix + '.csv')
