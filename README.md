# Activity Specification

Author: Lijun Yu

Email: lijun@lj-y.com

Submodule for specification and evaluation of activities in the [ActEV](https://actev.nist.gov) context.

## Activity Format

### Cube Activities

Spatial-temporal cube activities in a video, with optional confidence scores.
Each activity is represented as `(id, type, score, t0, t1, x0, y0, x1, y1)`, where type is based on an enum definition.

For class-agnostic proposals,

```python
import torch
from activity_spec import CubeActivities, CubeColumns, ProposalType

cubes = torch.zeros((10, len(CubeColumns)))  # 10 proposals
video_name = 'test.avi'
proposals = CubeActivities(cubes, video_name, ProposalType, CubeColumns)

proposals.to_internal()  # Convert to pandas.DataFrame to view in jupyter

save_dir = '.'
proposals.save(save_dir)  # Save as csv

load_dir = '.'   # Load from csv
proposals = CubeActivities.load(video_name, load_dir, ProposalType)
```

For activities,

```python
import torch
from activity_spec import ActivityTypes, CubeActivities, CubeColumns

cubes = torch.zeros((10, len(CubeColumns)))  # 10 activities
video_name = 'test.avi'
dataset = 'MEVA'
cube_acts = CubeActivities(
    cubes, video_name, ActivityTypes[dataset], CubeColumns)

cube_acts.to_official()  # Convert to official Json structure
```

For wrapped labels,

```python
import torch
from activity_spec import ActivityTypes, CubeActivities

video_name = 'test.avi'
dataset = 'MEVA'
labels = torch.zeros((10, len(ActivityTypes[dataset])))
wrapped_labels = CubeActivities(labels, video_name, None, ActivityTypes[dataset])

save_dir = '.'
wrapped_labels.save(save_dir)  # Save as csv

load_dir = '.'   # Load from csv
wrapped_labels = CubeActivities.load(video_name, load_dir, None)
```

See details at [cube.py](cube.py).

### Dataset

Generate a clip dataset based on proposals and labels. Number of negative samples can be controlled via the `negative_fraction` argument.

```python
from activity_spec import ProposalDataset
# Training: random sample negative proposals, drop ignored proposals
train_set = ProposalDataset(file_index_path, proposal_dir, label_dir,
                            video_dir, negative_fraction=1.)
# Testing: use all proposals
test_set = ProposalDataset(file_index_path, proposal_dir, label_dir,
                           video_dir, eval_mode=True)
clip, label = train_set[0]
```

See details at [dataset.py](dataset.py).

## Evaluations

### ActEV Scorer in Parallel

Run [ActEV_Scorer](https://github.com/usnistgov/ActEV_Scorer.git) with [actev-datasets](https://github.com/CMU-INF-DIVA/actev-datasets).
The ActEV_Scorer is called in parallel for each type of activity.

```sh
cd ..
python -m activity_spec.evaluate <dataset_dir> <subset_name> <prediction_file> <evaluation_dir>
# For example
python -m activity_spec.evaluate \
    actev-datasets/meva \
    kitware_eo_s2-test_99 \
    experiments/xxx/output.json \
    experiments/xxx/eval
```

### Proposal Evaluation

For each proposal generation model, implement `Matcher` class that matches proposal and ground truth. Then use it with [evaluate_proposal.py](evaluate_proposal.py).

For example,

```python
from activity_spec.evaluate_proposal import main, parse_args

class Matcher(object):

    def __init__(self, args):
        ...  # Initialize something with args

    def __call__(self, cube_acts_ref, cube_acts_det):
        ...  # Match proposals with ground truth and generate labels
        return cube_acts_labeled, wrapped_label_weights

if __name__ == "__main__":
    args = parse_args()
    matcher = Matcher(args)
    main(args, matcher)
```

## Dependency

See [actev_base](https://github.com/CMU-INF-DIVA/actev_base).

## License

See [License](LICENSE).
