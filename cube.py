import os.path as osp
from enum import EnumMeta

import numpy as np
import pandas as pd
import torch


class CubeActivities(object):

    '''
    Activities each as a spatial-temporal cube.
    cubes: [(type, x0, y0, x1, y1, t0, t1)] as torch.int.
    '''

    COLUMNS = ['type', 'x0', 'y0', 'x1', 'y1', 't0', 't1']

    def __init__(self, cubes: torch.Tensor, video_name: str,
                 type_names: EnumMeta):
        assert cubes.dim() == 2 and cubes.shape[1] == 7 and \
            cubes.dtype == torch.int, 'Proposal format invalid'
        self.cubes = cubes
        self.video_name = video_name
        self.type_names = type_names

    def to_internal(self):
        '''
        Internal storage format as pd.DataFrame.
        '''
        df = pd.DataFrame(self.cubes.cpu().numpy(), columns=self.COLUMNS)
        df['type'] = df['type'].apply(lambda v: self.type_names(v).name)
        return df

    def to_official(self):
        '''
        Official format in Json-like structure, 
        only contains temporal and type information.
        '''
        activities = []
        for cube_i in range(self.cubes.shape[0]):
            type_id, _, _, _, _, t0, t1 = self.cubes[cube_i].tolist()
            activity = {
                'activity': self.type_names(type_id).name,
                'localization': {self.video_name: {str(t0): 1, str(t1): 0}}}
            activities.append(activity)
        return activities

    def save(self, save_dir: str):
        df = self.to_internal()
        filename = self._get_internal_filename(self.video_name, save_dir)
        df.to_csv(filename)

    @classmethod
    def load(cls, video_name: str, load_dir: str, type_names: EnumMeta):
        filename = cls._get_internal_filename(video_name, load_dir)
        df = pd.read_csv(filename, index_col=0)
        df['type'] = df['type'].apply(lambda v: type_names[v].value)
        cubes = torch.as_tensor(df.values.astype(np.int), dtype=torch.int)
        return cls(cubes, video_name, type_names)

    @staticmethod
    def _get_internal_filename(video_name, data_dir):
        return osp.join(data_dir, osp.splitext(video_name)[0] + '.csv')
