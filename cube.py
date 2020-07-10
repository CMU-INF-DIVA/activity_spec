import os.path as osp

import pandas as pd
import torch


class CubeProposals(object):

    '''
    Proposals each as a spatial-temporal cube.
    cubes: [(x0, y0, x1, y1, t0, t1)] as torch.int.
    '''

    COLUMNS = ['x0', 'y0', 'x1', 'y1', 't0', 't1']

    def __init__(self, cubes: torch.Tensor, video_name: str):
        assert cubes.dim() == 2 and cubes.shape[1] == 6 and \
            cubes.dtype == torch.int, 'Proposal format invalid'
        self.cubes = cubes
        self.video_name = video_name

    def to_pandas(self):
        return pd.DataFrame(self.cubes.cpu().numpy(), columns=self.COLUMNS)

    def save(self, save_dir):
        df = self.to_pandas()
        filename = self._get_filename(self.video_name, save_dir)
        df.to_csv(filename)

    @classmethod
    def load(cls, video_name, load_dir):
        filename = cls._get_filename(video_name, load_dir)
        df = pd.read_csv(filename, index_col=0)
        cubes = torch.as_tensor(df.values, dtype=torch.int)
        return cls(cubes, video_name)

    @staticmethod
    def _get_filename(video_name, data_dir):
        return osp.join(data_dir, osp.splitext(video_name)[0] + '.csv')
