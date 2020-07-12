import argparse
import gzip
import json
import os
import os.path as osp
import subprocess
from collections import defaultdict, namedtuple
from multiprocessing.pool import Pool

import pandas as pd
from pyturbo import get_logger, progressbar

SCORER = osp.join(osp.dirname(__file__), 'scorer/ActEV_Scorer.py')
NAME = '%s.%s' % (__package__, osp.splitext(osp.basename(__file__))[0])

Job = namedtuple('Job', [
    'activity_type', 'evaluation_dir', 'protocol',
    'file_index_path', 'activity_index_dir',
    'file_list', 'reference_activities', 'prediction_activities'])

logger = get_logger(NAME)


def group_activities_by_type(activities):
    activities_by_type = defaultdict(list)
    for activity in activities:
        activity_type = activity['activity']
        activity = activity.copy()
        activity.pop('objects', None)
        activities_by_type[activity_type].append(activity)
    return activities_by_type


def activity_worker(job):
    evaluation_dir = osp.join(job.evaluation_dir, job.activity_type)
    os.makedirs(evaluation_dir, exist_ok=True)
    activity_index_path = osp.join(
        job.activity_index_dir, '%s.json' % (job.activity_type))
    reference_path = osp.join(evaluation_dir, 'reference.json')
    prediction_path = osp.join(evaluation_dir, 'prediction.json')
    reference = {'filesProcessed': job.file_list,
                 'activities': job.reference_activities}
    prediction = {'filesProcessed': job.file_list,
                  'activities': job.prediction_activities}
    with open(reference_path, 'w') as f:
        json.dump(reference, f, indent=4)
    with open(prediction_path, 'w') as f:
        json.dump(prediction, f, indent=4)
    cmd = f'python {SCORER} {job.protocol} -a {activity_index_path} ' \
        f'-f {job.file_index_path} -r {reference_path} -s {prediction_path} ' \
        f'-o {evaluation_dir} -v -n {os.cpu_count()}'
    process = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
    if process.returncode != 0:
        logger.error('Scorer process failed for type %s: \n%s' % (
            job.activity_type, process.stdout.decode('utf-8')))
        raise ValueError(process.returncode)
    metrics = pd.read_csv(
        osp.join(evaluation_dir, 'scores_by_activity.csv'), '|')
    metrics = metrics[['metric_name', 'metric_value']]
    metrics = metrics.set_index('metric_name')
    metrics['metric_value'] = metrics['metric_value'].apply(
        lambda x: x if x != 'None' else 0).astype(float)
    metrics.columns = [job.activity_type]
    return metrics


def main(args):
    logger.info('Running with args: \n\t%s', '\n\t'.join([
                '%s = %s' % (k, v) for k, v in vars(args).items()]))
    os.makedirs(args.evaluation_dir, exist_ok=True)
    file_index_path = osp.join(
        args.dataset_dir, 'meta/file-index/%s.json' % (args.subset))
    activity_index_dir = osp.join(
        args.dataset_dir, 'meta/activity-index/single')
    reference_path = osp.join(
        args.dataset_dir, 'meta/reference/%s.json.gz' % (args.subset))
    logger.info('Using file index: %s', file_index_path)
    logger.info('Loading reference: %s', reference_path)
    with gzip.open(reference_path, 'rt', encoding='utf-8') as f:
        reference = json.load(f)
    logger.info('Loading prediction: %s', args.prediction_file)
    with open(args.prediction_file) as f:
        prediction = json.load(f)
    assert set(reference['filesProcessed']) == set(
        prediction['filesProcessed']), 'File list does not match'
    file_list = reference['filesProcessed']
    logger.info('Grouping activities by type')
    reference_by_type = group_activities_by_type(reference['activities'])
    prediction_by_type = group_activities_by_type(prediction['activities'])
    logger.info('Evaluating')
    jobs = []
    for activity_type in reference_by_type.keys():
        job = Job(
            activity_type, args.evaluation_dir, args.protocol,
            file_index_path, activity_index_dir, file_list,
            reference_by_type[activity_type],
            prediction_by_type[activity_type])
        jobs.append(job)
    with Pool() as pool:
        metrics = [*progressbar(pool.imap_unordered(activity_worker, jobs),
                                'Evaluation by type', total=len(jobs))]
    metrics = sorted(metrics, key=lambda x: x.columns[0])
    metrics = pd.concat(metrics, axis=1)
    metrics.to_csv(osp.join(args.evaluation_dir, 'metrics_by_activity.csv'))
    metrics_mean = metrics.mean(axis=1).to_frame()
    metrics_mean.columns = ['mean']
    metrics_mean.to_csv(osp.join(args.evaluation_dir, 'metrics.csv'))
    keys = ['nAUDC@0.2tfa', 'p_miss@0.04tfa']
    logger.info('Metrics: \n\t%s', '\n\t'.join(['%s = %.4f' % (
        key, metrics_mean.loc[key, 'mean']) for key in keys]))


def parse_args(argv=None):
    parser = argparse.ArgumentParser('python -m %s' % (NAME))
    parser.add_argument(
        'dataset_dir', help='Path of dataset (e.g. actev_datasets/meva)')
    parser.add_argument('subset', help='Name of subset')
    parser.add_argument('prediction_file', help='Path of system output')
    parser.add_argument('evaluation_dir', help='Path of evaluation results')
    parser.add_argument(
        '--protocol', default='ActEV_SDL_V2', help='Scorer protocol')
    args = parser.parse_args(argv)
    return args


if __name__ == "__main__":
    main(parse_args())
