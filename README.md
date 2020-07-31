# Activity-spec

Author: Lijun Yu

Email: lijun@lj-y.com

Submodule for specification and evaluation of activities in the [ActEV](https://actev.nist.gov) context.

## Activity Format

### Cube Activities

Spatial-temporal cube activities in a video, with optional confidence scores.
Each activity is represented as `(type, score, t0, t1, x0, y0, x1, y1)`, where type is based on an enum definition.

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
proposals = CubeActivities.load(video_name, load_dir, ProposalType, CubeColumns)
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

See details [cube.py](cube.py).

## Scorer

Run [ActEV_Scorer](https://github.com/usnistgov/ActEV_Scorer.git) with [actev-datasets](https://github.com/CMU-INF-DIVA/actev-datasets).
The ActEV_Scorer is called in parallel for each type of activity.

```sh
cd ..
python -m activity_spec.evaluate <dataset_dir> <subset_name> <prediction_file> <evaluation_dir>
# For example
python -m activity_spec.evaluate \
    actev-datasets/meva \
    kitware_eo_s1-train_171 \
    experiments/xxx/output.json \
    experiments/xxx/eval
```

## Dependency

See [actev_base](https://github.com/CMU-INF-DIVA/actev_base).

## License

See [License](LICENSE).
