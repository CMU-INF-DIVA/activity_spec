import gzip
import json

from .base import ActivityType


class Reference(object):

    def __init__(self, reference_path):
        with gzip.open(reference_path, 'rt', encoding='utf-8') as f:
            reference = json.load(f)
        self.video_list = reference['filesProcessed']
        self.activities = self.split_by_video(reference['activities'])

    def split_by_video(self, activities):
        activities_by_video = {v: [] for v in self.video_list}
        for activity in activities:
            video_name = [*activity['localization'].keys()][0]
            activities_by_video[video_name].append(activity)
        return activities_by_video

    def get_quantized_cubes(self, video_name, stride=32):
        activities = self.activities[video_name]
        quantized_activities = []
