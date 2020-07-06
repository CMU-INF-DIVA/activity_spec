import os.path as osp
from typing import List

from .base import Collection, Proposal


class VideoCollection(Collection):

    '''
    A collection for proposals in a video.
    '''

    def __init__(self, proposals: List[Proposal], video_name: str):
        super(VideoCollection, self).__init__(proposals)
        self.video_name = video_name

    def save(self, save_dir):
        filename = self.get_filename(self.video_name, save_dir)
        super(VideoCollection, self).save_record(filename)

    @classmethod
    def load(cls, video_name, load_dir):
        filename = cls.get_filename(video_name, load_dir)
        record = super(VideoCollection, cls).load_record(filename)
        proposals = super(VideoCollection, cls).parse_record(record)
        return cls(proposals, video_name)

    @staticmethod
    def get_filename(video_name, data_dir):
        return osp.join(data_dir, osp.splitext(video_name)[0] + '.csv')
