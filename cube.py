import os.path as osp
from enum import EnumMeta
from typing import Tuple

import numpy as np
import pandas as pd
import torch


class CubeActivities(object):

    '''
    Activities each as a spatial-temporal cube.
    cubes: [(type, score, t0, t1, x0, y0, x1, y1)].
    '''

    CUBE_COLUMNS = ['type', 'score', 't0', 't1', 'x0', 'y0', 'x1', 'y1']

    def __init__(self, cubes: torch.Tensor, video_name: str,
                 type_names: EnumMeta):
        assert cubes.ndim == 2 and cubes.shape[1] == 8, \
            'Proposal format invalid'
        self.cubes = cubes
        self.video_name = video_name
        self.type_names = type_names

    def to_internal(self):
        '''
        Internal storage format as pd.DataFrame.
        '''
        df = pd.DataFrame(self.cubes.cpu().numpy(), columns=self.CUBE_COLUMNS)
        df['type'] = df['type'].apply(lambda v: self.type_names(v).name)
        return df

    def to_official(self):
        '''
        Official format in Json structure, 
        only contains temporal and type information.
        '''
        activities = []
        for cube_i in range(self.cubes.shape[0]):
            type_id, score, t0, t1, _, _, _, _ = self.cubes[cube_i].tolist()
            activity = {
                'activity': self.type_names(type_id).name,
                'localization': {
                    self.video_name: {str(int(t0)): 1, str(int(t1)): 0}},
                'presenceConf': score}
            activities.append(activity)
        return activities

    def save(self, save_dir: str):
        '''
        Save as csv file in save_dir.
        '''
        df = self.to_internal()
        filename = self._get_internal_filename(self.video_name, save_dir)
        df.to_csv(filename)

    @classmethod
    def load(cls, video_name: str, load_dir: str, type_names: EnumMeta):
        '''
        Load from csv file in load_dir.
        '''
        filename = cls._get_internal_filename(video_name, load_dir)
        df = pd.read_csv(filename, index_col=0)
        df['type'] = df['type'].apply(lambda v: type_names[v].value)
        cubes = torch.as_tensor(df[cls.CUBE_COLUMNS].values.astype(np.float32))
        obj = cls(cubes, video_name, type_names)
        return obj

    def spatial_enlarge(self, enlarge_rate: float, spatial_limit: Tuple):
        '''
        Enlarge spatial boxes.
        enlarge_rate: enlarge rate in each axis.
        spatial_limit: (x, y), frame size.
        '''
        spatial_size = self.cubes[:, 6:8] - self.cubes[:, 4:6]
        enlarge_size = spatial_size * enlarge_rate
        new_cubes = self.cubes.clone()
        new_cubes[:, 4:6] = torch.clamp(
            self.cubes[:, 4:6] - enlarge_size, min=0)
        new_cubes[:, 6:8] = torch.min(
            self.cubes[:, 6:8] + enlarge_size,
            torch.as_tensor([spatial_limit], dtype=torch.float))
        return CubeActivities(new_cubes, self.video_name, self.type_names)

    @staticmethod
    def _get_internal_filename(video_name, data_dir):
        return osp.join(data_dir, osp.splitext(video_name)[0] + '.csv')
