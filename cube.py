import os.path as osp
from enum import EnumMeta
from typing import Union

import numpy as np
import pandas as pd
import torch


class CubeActivities(object):

    '''
    Activities each as a spatial-temporal cube.
    cubes: [(type, x0, y0, x1, y1, t0, t1)] as torch.int.
    '''

    CUBE_COLUMNS = ['type', 'x0', 'y0', 'x1', 'y1', 't0', 't1']

    def __init__(self, cubes: torch.Tensor, video_name: str,
                 type_names: EnumMeta):
        assert cubes.ndim == 2 and cubes.shape[1] == 7 and \
            cubes.dtype == torch.int, 'Proposal format invalid'
        self.cubes = cubes
        self.video_name = video_name
        self.type_names = type_names
        self.scores = None

    def set_scores(self, scores: torch.Tensor):
        assert scores.ndim == 1 and scores.shape[0] == self.cubes.shape[0] and \
            scores.dtype == torch.float, 'Score length inconsistent with cubes'
        self.scores = scores

    def to_internal(self):
        '''
        Internal storage format as pd.DataFrame.
        '''
        df = pd.DataFrame(self.cubes.cpu().numpy(), columns=self.CUBE_COLUMNS)
        df['type'] = df['type'].apply(lambda v: self.type_names(v).name)
        if self.scores is not None:
            df['score'] = self.scores.cpu().numpy()
        return df

    def to_official(self):
        '''
        Official format in Json structure, 
        only contains temporal and type information.
        '''
        activities = []
        for cube_i in range(self.cubes.shape[0]):
            type_id, _, _, _, _, t0, t1 = self.cubes[cube_i].tolist()
            score = self.scores[cube_i] if self.scores is not None else 1
            activity = {
                'activity': self.type_names(type_id).name,
                'localization': {self.video_name: {str(t0): 1, str(t1): 0}},
                'presenceConf': score}
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
        cubes = torch.as_tensor(
            df[cls.CUBE_COLUMNS].values.astype(np.int), dtype=torch.int)
        obj = cls(cubes, video_name, type_names)
        if 'score' in df:
            obj.set_scores(torch.as_tensor(
                df['score'].values.astype(np.float32), dtype=torch.float))
        return obj

    @staticmethod
    def _get_internal_filename(video_name, data_dir):
        return osp.join(data_dir, osp.splitext(video_name)[0] + '.csv')
