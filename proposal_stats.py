import argparse
import json
import os.path as osp

import pandas as pd
from pyturbo import get_logger

NAME = '%s.%s' % (__package__, osp.splitext(osp.basename(__file__))[0])

logger = get_logger(NAME)


def dataframe_to_markdown(df, name):
    df = df.apply(lambda l: l.apply(lambda v: '%.04f' % v))
    df.index.name = name
    s = df.to_markdown()
    s = s.replace('|-', '|:').replace('-|', ':|')
    return s


def main(args):
    logger.info('Running with args: \n\t%s', '\n\t'.join([
                '%s = %s' % (k, v) for k, v in vars(args).items()]))
    df = pd.read_csv(osp.join(args.proposal_evaluation_dir, 'metrics.csv'),
                     index_col=0)
    with open(osp.join(args.proposal_evaluation_dir, 'stats.json')) as f:
        stats = json.load(f)
    row = {'#Proposal': stats['cube_counts']['detection'],
           '#Cubed reference': stats['cube_counts']['reference'],
           '%Positive': stats['cube_rates']['positive / detection'],
           '%(#label <= 1)': stats['label_rates']['1'],
           '%(#label <= 2)': stats['label_rates']['1'] +
           stats['label_rates']['2'],
           '%(#label <= 3)': stats['label_rates']['1'] +
           stats['label_rates']['2'] + stats['label_rates']['3']}
    stats_df = pd.DataFrame([row], index=[args.subset_type])
    metrics = ['nAUDC@0.2tfa', 'p_miss@0.04tfa']
    metrics_df = df.loc[metrics]
    sub_dfs = []
    for mode in ['IoU', 'RefCover']:
        columns = ['%s_avg' % (mode)] + [
            '%s_0.%d' % (mode, i) for i in range(10)]
        sub_df = metrics_df[columns]
        sub_df.columns = ['Average'] + [
            '>= 0.%d' % (i) for i in range(10)]
        sub_df.index = ['%s (%s)' % (idx, mode)
                        for idx in sub_df.index]
        sub_dfs.append(sub_df)
    metrics_df = pd.concat(sub_dfs)
    print(dataframe_to_markdown(stats_df, 'Statistics'), '\n')
    print(dataframe_to_markdown(metrics_df, args.subset_type), '\n')


def parse_args(argv=None):
    parser = argparse.ArgumentParser('python -m %s' % (NAME))
    parser.add_argument(
        'proposal_evaluation_dir',
        help='Directory containing proposal evaluation results')
    parser.add_argument('subset_type', choices=['train', 'test', 'all'])
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    main(parse_args())
