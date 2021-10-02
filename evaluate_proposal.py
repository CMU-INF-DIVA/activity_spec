import argparse
import json
import os
import os.path as osp
from collections import Counter, namedtuple
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import psutil
import torch
from pyturbo import get_logger, progressbar
from torchvision.ops.boxes import box_area, box_iou

from .assigner import ActivityAssigner
from .base import ActivityTypes, ProposalType
from .cube import CubeActivities
from .evaluate import METRIC_KEYS
from .evaluate import main as evaluate_main
from .evaluate import parse_args as evaluate_args
from .reference import Reference

NAME = '%s.%s' % (__package__, osp.splitext(osp.basename(__file__))[0])
MODES = ['IoU', 'RefCover']
THRESHOLDS = np.arange(0, 1, 0.1)

Job = namedtuple('Job', [
    'reference', 'mode', 'threshold', 'subset', 'target', 'dataset_dir',
    'evaluation_dir', 'num_processes'])


logger = get_logger(NAME)


def get_spatial_scores(cube_acts_ref, cube_acts_prop):
    if len(cube_acts_ref) == 0 or len(cube_acts_prop) == 0:
        scores = torch.zeros((len(cube_acts_prop,)))
        spatial_scores = {'IoU': scores, 'RefCover': scores}
        return spatial_scores
    ref_time = cube_acts_ref.cubes[:, cube_acts_ref.columns.t0].type(torch.int)
    prop_time = cube_acts_prop.cubes[
        :, cube_acts_prop.columns.t0].type(torch.int)
    temporal = ref_time.unsqueeze(1) == prop_time.unsqueeze(0)
    boxes_ref = cube_acts_ref.cubes[
        :, cube_acts_ref.columns.x0:cube_acts_ref.columns.y1 + 1]
    boxes_det = cube_acts_prop.cubes[
        :, cube_acts_prop.columns.x0:cube_acts_prop.columns.y1 + 1]
    iou = box_iou(boxes_ref, boxes_det)
    area_ref = box_area(boxes_ref)[:, None]
    left_top = torch.max(boxes_ref[:, None, :2], boxes_det[:, :2])
    right_bottom = torch.min(boxes_ref[:, None, 2:], boxes_det[:, 2:])
    width_height = (right_bottom - left_top).clamp(min=0)
    area_inter = width_height[:, :, 0] * width_height[:, :, 1]
    cover = area_inter / area_ref
    spatial_scores = {'IoU': (iou * temporal).max(axis=0).values,
                      'RefCover': (cover * temporal).max(axis=0).values}
    return spatial_scores


def threshold_worker(job):
    reference = job.reference
    reference['activities'] = [act for act in reference['activities']
                               if act[job.mode] >= job.threshold]
    labeled_prop_path = osp.join(
        job.evaluation_dir, 'labeled_prop_%s_%.1f.json' % (
            job.mode, job.threshold))
    with open(labeled_prop_path, 'w') as f:
        json.dump(reference, f, indent=4)
    evaluation_dir = osp.join(
        job.evaluation_dir, 'eval_%s_%.1f' % (job.mode, job.threshold))
    argv = [job.dataset_dir, job.subset, labeled_prop_path,
            evaluation_dir, '--target', job.target, '--silent',
            '--num_processes', str(job.num_processes)]
    current_metric = evaluate_main(evaluate_args(argv))
    current_metric.columns = ['%s_%.1f' % (job.mode, job.threshold)]
    return current_metric


def main(args, assigner=None):
    logger.info('Running with args: \n\t%s', '\n\t'.join([
                '%s = %s' % (k, v) for k, v in vars(args).items()]))
    dataset_dir = osp.join(args.datasets_dir, args.dataset)
    reference_path = osp.join(
        dataset_dir, 'meta/reference/%s.json.gz' % (args.subset))
    logger.info('Loading reference: %s', reference_path)
    reference = Reference(reference_path, ActivityTypes[args.dataset])
    if sum([len(v) for v in reference.activities.values()]) == 0:
        return
    os.makedirs(args.evaluation_dir, exist_ok=True)
    logger.info('Loading proposals: %s', args.proposal_dir)
    logger.info('Loading labels: %s', args.label_dir)
    if assigner is None:
        assigner = ActivityAssigner()
    all_activities = []
    pos_count, ref_count, det_count = 0, 0, 0
    num_labels = []
    for video in progressbar(reference.video_list, 'Videos'):
        cube_acts_ref = reference.get_quantized_cubes(
            video, args.duration, args.stride)
        cube_acts_prop = CubeActivities.load(
            video, args.proposal_dir, ProposalType)
        wrapped_label_weights = CubeActivities.load(
            video, args.label_dir, None)
        cube_acts_labeled = assigner(cube_acts_prop, wrapped_label_weights)
        if args.enlarge_rate is not None:
            cube_acts_prop = cube_acts_prop.spatial_enlarge(
                args.enlarge_rate, args.frame_size)
        n_labels = (wrapped_label_weights.cubes[:, 1:] > 0).sum(dim=1)
        pos_count += (n_labels > 0).sum().item()
        ref_count += len(cube_acts_ref)
        det_count += len(cube_acts_prop)
        num_labels.append(n_labels)
        spatial_scores = get_spatial_scores(cube_acts_ref, cube_acts_labeled)
        activities = cube_acts_labeled.to_official()
        for key, value in spatial_scores.items():
            for act, v in zip(activities, value.tolist()):
                act[key] = v
        all_activities.extend(activities)
    for i, act in enumerate(all_activities):
        act['activityID'] = i
    labeled_prop = {'filesProcessed': reference.video_list,
                    'activities': all_activities}
    logger.info('Gather statistics')
    num_labels = Counter(torch.cat(num_labels).tolist())
    label_count = sum([k * v for k, v in num_labels.items()])
    stats = {'cube_rates': {'positive / detection': pos_count / det_count,
                            'positive / reference': pos_count / ref_count,
                            'label / reference': label_count / ref_count},
             'label_rates': {k: v / pos_count for k, v in num_labels.items()
                             if k > 0},
             'cube_counts': {'positive': pos_count, 'detection': det_count,
                             'reference': ref_count},
             'label_counts': {'total': label_count, **num_labels}}
    with open(osp.join(args.evaluation_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)
    logger.info('Statistics: \n%s', json.dumps(stats, indent=4))
    logger.info('Evaluating')
    num_processes = len(psutil.Process().cpu_affinity()) // args.num_processes
    jobs = []
    for threshold in THRESHOLDS:
        for mode in MODES:
            job = Job(labeled_prop, mode, threshold, args.subset, args.target,
                      dataset_dir, args.evaluation_dir, num_processes)
            jobs.append(job)
    with ProcessPoolExecutor(args.num_processes) as pool:
        metrics = [*progressbar(
            pool.map(threshold_worker, reversed(jobs)),
            'Jobs', total=len(jobs))]
    metrics = pd.concat(metrics, axis=1)
    pd.set_option('max_columns', None)
    print_keys = METRIC_KEYS[args.target]
    all_columns = []
    for mode in MODES:
        columns = ['%s_%.1f' % (mode, thres) for thres in THRESHOLDS]
        metrics['%s_avg' % (mode)] = metrics[columns].mean(axis=1)
        columns.insert(0, '%s_avg' % (mode))
        logger.info('Metrics in mode %s: \n%s', mode,
                    metrics.loc[print_keys, columns])
        all_columns.extend(columns)
    metrics = metrics[all_columns]
    metrics.to_csv(osp.join(args.evaluation_dir, 'metrics.csv'))
    return metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser('python -m %s' % (NAME))
    parser.add_argument(
        'subset', help='Name of subset (e.g. kitware_eo_s2-train_158)')
    parser.add_argument(
        'proposal_dir', help='Directory containing proposals')
    parser.add_argument('label_dir', help='Directory containing labels')
    parser.add_argument(
        'evaluation_dir', help='Directory to store evaluation results')
    parser.add_argument(
        '--dataset', default='MEVA', choices=[*ActivityTypes.keys()],
        help='Dataset name')
    parser.add_argument(
        '--duration', default=64, type=int,
        help='Duration of a proposal (default: 64 frames)')
    parser.add_argument(
        '--stride', default=16, type=int,
        help='Stride between proposals (default: 16 frames)')
    parser.add_argument(
        '--enlarge_rate', default=None, type=float,
        help='Spatial enlarge rate of proposal')
    parser.add_argument(
        '--frame_size', default=[1920, 1080], type=int, nargs=2,
        help='Image size (width, height) of a frame, (default: [1920, 1080])')
    parser.add_argument(
        '--target', default='SDL', choices=METRIC_KEYS.keys(),
        help='Evaluation target, only affects the metrics to be printed')
    parser.add_argument(
        '--datasets_dir', help='Directory of datasets (actev-datasets repo)',
        default=osp.join(osp.dirname(__file__), '../../datasets'))
    parser.add_argument(
        '--num_processes', type=int,
        default=min(len(psutil.Process().cpu_affinity()),
                    len(MODES) * len(THRESHOLDS)),
        help='Number of processes')
    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    main(parse_args())
