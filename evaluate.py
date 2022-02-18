import argparse
import gzip
import json
import logging
import os
import os.path as osp
import signal
import subprocess
import sys
import tempfile
from collections import defaultdict, namedtuple

import pandas as pd
import psutil
from pyturbo import get_logger, process_map

SCORER = osp.join(osp.dirname(__file__), 'scorer/ActEV_Scorer.py')
NAME = '%s.%s' % (__package__, osp.splitext(osp.basename(__file__))[0])

Job = namedtuple('Job', [
    'activity_type', 'evaluation_dir', 'protocol',
    'file_index_path', 'activity_index_dir',
    'file_list', 'reference_activities', 'prediction_activities',
    'max_false_activity_length', 'scorer_num_processes'])
METRIC_KEYS = {
    'SDL': ('ActEV_SDL_V2', [
        'nAUDC@0.2tfa', 'p_miss@0.04tfa', 'p_miss@0.02tfa']),
    'TRECVID': ('ActEV_SDL_V2', [
        'nAUDC@0.2tfa', 'p_miss@0.15tfa', 'w_p_miss@0.15rfa']),
    'SRL': ('SRL_AD_V1', [
        'nAUDC@0.2rfa', 'p_miss@0.1rfa']),
    'SRL_AOD': ('SRL_AOD_V1', [
        'nAUDC@0.2rfa', 'p_miss@0.1rfa', 'n-mode@0.1rfa']),
}

logger = get_logger(NAME.split('.')[-1])


def group_activities_by_type(activities, keep_objects):
    activities_by_type = defaultdict(list)
    for activity in activities:
        activity_type = activity['activity']
        if not keep_objects:
            activity = activity.copy()
            activity.pop('objects', None)
        activities_by_type[activity_type].append(activity)
    return activities_by_type


def worker_initializer():
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


def get_prediction(job, reference_length):
    sorted_activities = sorted(
        job.prediction_activities, key=lambda a: a['presenceConf'],
        reverse=True)
    if job.max_false_activity_length is not None:
        selected_activities = []
        max_activity_length = reference_length + job.max_false_activity_length
        current_length = 0
        for act in sorted_activities:
            frame_id_1, frame_id_2 = [
                *[*act['localization'].values()][0].keys()]
            length = abs(int(frame_id_1) - int(frame_id_2))
            current_length += length
            selected_activities.append(act)
            if current_length > max_activity_length:
                break
    else:
        selected_activities = sorted_activities
    status = {'status': 'success', 'message': ''}
    prediction = {
        'filesProcessed': job.file_list,
        'processingReport': {
            'siteSpecific': {},
            'fileStatuses': {f: status for f in job.file_list}},
        'activities': selected_activities}
    return prediction


def activity_worker(job):
    evaluation_dir = osp.join(job.evaluation_dir, job.activity_type)
    os.makedirs(evaluation_dir, exist_ok=True)
    activity_index_path = osp.join(
        job.activity_index_dir, '%s.json' % (job.activity_type))
    reference = {'filesProcessed': job.file_list,
                 'activities': job.reference_activities}
    reference_length = 0
    for act in job.reference_activities:
        loc = {v: int(k)
               for k, v in [*act['localization'].values()][0].items()}
        reference_length += loc[0] - loc[1]
    reference_path = osp.join(evaluation_dir, 'reference.json')
    with open(reference_path, 'w') as f:
        json.dump(reference, f, indent=4)
    prediction = get_prediction(job, reference_length)
    prediction_path = osp.join(evaluation_dir, 'prediction.json')
    with open(prediction_path, 'w') as f:
        json.dump(prediction, f, indent=4)
    cmd = f'{sys.executable} {SCORER} {job.protocol} ' \
        f'-a {activity_index_path} -f {job.file_index_path} ' \
        f'-r {reference_path} -s {prediction_path} -o {evaluation_dir} ' \
        f'--det-point-resolution 1024 -v -n {job.scorer_num_processes}'
    columns = [job.activity_type]
    del reference, prediction, job
    try:
        subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as e:
        print('\n' + e.stdout.decode('utf-8'))
        raise e
    metrics = pd.read_csv(
        osp.join(evaluation_dir, 'scores_by_activity.csv'), sep='|')
    metrics = metrics[['metric_name', 'metric_value']]
    metrics = metrics.set_index('metric_name')
    metrics['metric_value'] = metrics['metric_value'].apply(
        lambda x: x if x != 'None' else 0).astype(float)
    metrics.columns = columns
    return metrics


def main(args):
    if args.silent:
        logger.setLevel(logging.WARN)
    logger.info('Running with args: \n\t%s', '\n\t'.join([
                '%s = %s' % (k, v) for k, v in vars(args).items()]))
    os.makedirs(args.evaluation_dir, exist_ok=True)
    file_index_path = osp.join(
        args.dataset_dir, 'meta/file-index/%s.json' % (args.subset))
    activity_index_dir = osp.join(args.dataset_dir, 'meta/activity-index')
    single_activity_index_dir = osp.join(activity_index_dir, 'single')
    reference_path = osp.join(
        args.dataset_dir, 'meta/reference/%s.json.gz' % (args.subset))
    logger.info('Loading file index: %s', file_index_path)
    with open(file_index_path) as f:
        file_index = json.load(f)
    video_length = 0
    for attributes in file_index.values():
        length = {v: int(k) for k, v in attributes['selected'].items()}[0] - 1
        video_length += length
    logger.info('Video length: %d frames', video_length)
    logger.info('Loading prediction: %s', args.prediction_file)
    with open(args.prediction_file) as f:
        prediction = json.load(f)
    logger.info('Loading reference: %s', reference_path)
    with gzip.open(reference_path, 'rt', encoding='utf-8') as f:
        reference = json.load(f)
    assert set(reference['filesProcessed']) == set(
        prediction['filesProcessed']), 'File list does not match'
    file_list = reference['filesProcessed']
    logger.info('Grouping activities by type')
    keep_objects = 'AOD' in args.target
    reference_by_type = group_activities_by_type(
        reference['activities'], keep_objects)
    prediction_by_type = group_activities_by_type(
        prediction['activities'], keep_objects)
    protocol, metric_keys = METRIC_KEYS[args.target]
    if protocol == 'ActEV_SDL_V2':
        logger.info('Pruning prediction at TFA threshold %.2f',
                    args.tfa_threshold)
        max_false_activity_length = video_length * args.tfa_threshold
    else:
        max_false_activity_length = None
    activity_types = [*reference_by_type.keys()]
    if args.activity_index_name is not None:
        logger.info('Using activity index %s', args.activity_index_name)
        with open(osp.join(activity_index_dir,
                           f'{args.activity_index_name}.json')) as f:
            activity_types_new = [*json.load(f).keys()]
        activity_types = list(set(activity_types) & set(activity_types_new))
    logger.info('Evaluating')
    num_workers = min(args.num_processes, len(activity_types))
    scorer_num_processes = max(1, args.num_processes // num_workers)
    jobs = []
    with tempfile.TemporaryDirectory() as evaluation_dir:
        if args.save_all:
            evaluation_dir = args.evaluation_dir
        for activity_type in activity_types:
            job = Job(
                activity_type, evaluation_dir, protocol,
                file_index_path, single_activity_index_dir, file_list,
                reference_by_type[activity_type],
                prediction_by_type[activity_type],
                max_false_activity_length, scorer_num_processes)
            jobs.append(job)
        metrics = [*process_map(
            activity_worker, jobs, num_workers,
            desc='Evaluation by type', silent=args.silent)]
    metrics = sorted(metrics, key=lambda x: x.columns[0])
    metrics = pd.concat(metrics, axis=1)
    metrics.to_csv(osp.join(args.evaluation_dir, 'metrics_by_activity.csv'))
    metrics_mean = metrics.mean(axis=1).to_frame()
    metrics_mean.columns = ['mean']
    metrics_mean.to_csv(osp.join(args.evaluation_dir, 'metrics.csv'))
    logger.info('Metrics: \n\t%s', '\n\t'.join(['%s = %.4f' % (
        key, metrics_mean.loc[key, 'mean']) for key in metric_keys]))
    return metrics_mean


def parse_args(argv=None):
    parser = argparse.ArgumentParser('python -m %s' % (NAME))
    parser.add_argument(
        'dataset_dir', help='Directory of dataset (e.g. actev_datasets/MEVA)')
    parser.add_argument(
        'subset', help='Name of subset (e.g. kitware_eo_s2-train_158)')
    parser.add_argument('prediction_file', help='Path of system output')
    parser.add_argument(
        'evaluation_dir', help='Directory of evaluation results')
    parser.add_argument(
        '--target', default='SDL', choices=METRIC_KEYS.keys(),
        help='Evaluation target, only affects the metrics to be printed')
    parser.add_argument(
        '--activity_index_name',
        help='Activity index name to select a subset of activity types')
    parser.add_argument(
        '--tfa_threshold', type=float, default=0.4,
        help='Time-based false alarm threshold to filter redundant instances')
    parser.add_argument('--silent', action='store_true', help='Silent logs')
    parser.add_argument(
        '--num_processes', type=int,
        default=len(psutil.Process().cpu_affinity()),
        help='Number of processes')
    parser.add_argument(
        '--save_all', action='store_true', help='Store intermediate results')
    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    main(parse_args())
