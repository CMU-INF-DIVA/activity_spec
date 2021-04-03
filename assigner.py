import torch

from .base import PROP_TYPE_FRAC

class ActivityAssigner(object):

    def __init__(self, threshold=0):
        self.threshold = threshold

    def __call__(self, cube_acts_prop, wrapped_label_weights):
        prop_indices, predictions = torch.nonzero(
            wrapped_label_weights.cubes > self.threshold, as_tuple=True)
        valid = predictions > 0
        prop_indices, predictions = prop_indices[valid], predictions[valid]
        cubes = cube_acts_prop.cubes[prop_indices]
        cubes[:, cube_acts_prop.columns.id] += cubes[
            :, cube_acts_prop.columns.type] * PROP_TYPE_FRAC
        cubes[:, cube_acts_prop.columns.type] = predictions
        cubes[:, cube_acts_prop.columns.score] = wrapped_label_weights.cubes[
            prop_indices, predictions]
        cube_acts = cube_acts_prop.duplicate_with(
            cubes, type_names=wrapped_label_weights.columns)
        return cube_acts
