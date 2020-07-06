from typing import List, Union

import torch

from .base import Proposal, ProposalRegistry


class CubeProposal(Proposal):

    '''
    A proposal as a spatial-temporal cube.
    spatial_box: (x0, y0, x1, y1).
    temporal_slice: (t0, t1) in frame id.
    '''

    RECORD = ['type', 'x0', 'y0', 'x1', 'y1', 't0', 't1']

    def __init__(self, spatial_box: torch.Tensor, temporal_slice: torch.Tensor):
        self.spatial_box = spatial_box
        self.temporal_slice = temporal_slice

    def spatial_enlarge(self, enlarge_rate: float, spatial_limit: torch.Tensor):
        '''
        Enlarge spatial box.
        enlarge_rate: enlarge rate in each axis.
        spatial_limit: (x, y), frame size.
        '''
        spatial_size = self.spatial_box[2:] - self.spatial_box[:2]
        enlarge_size = spatial_size * enlarge_rate
        new_spatial_box = torch.empty_like(self.spatial_box)
        new_spatial_box[:2] = torch.clamp(
            self.spatial_box[:2] - enlarge_size, min=0)
        new_spatial_box[2:] = torch.min(
            self.spatial_box[2:] + enlarge_size, spatial_limit)
        return CubeProposal(new_spatial_box, self.temporal_slice)

    def spatial_merge(self, another_proposal):
        new_spatial_box = torch.empty_like(self.spatial_box)
        new_spatial_box[:2] = torch.min(
            self.spatial_box[:2], another_proposal.spatial_box[:2])
        new_spatial_box[2:] = torch.max(
            self.spatial_box[2:], another_proposal.spatial_box[2:])
        return CubeProposal(new_spatial_box, self.temporal_slice)

    def get_cropped_frames(self, clip_frames) -> torch.Tensor:
        x0, x1, y0, y1 = self.spatial_box.round().astype(int)
        cropped_frames = clip_frames[:, y0:y1, x0:x1]
        return cropped_frames

    def to_record(self):
        record = [self.__class__.__name__, *self.spatial_box.tolist(),
                  *self.temporal_slice.tolist()]
        record = dict(zip(self.RECORD, record))
        return record

    @classmethod
    def from_record(cls, record):
        values = super(CubeProposal, cls).parse_record(record)
        tensor = torch.as_tensor(values)
        spatial_box = tensor[:4]
        temporal_slice = tensor[4:6]
        return cls(spatial_box, temporal_slice)


ProposalRegistry[CubeProposal.__name__] = CubeProposal
