# Xiaoya-Core

xiaoya 2.0 core

## Project Structure

```bash
xiaoya/ # root
    pyehr/ # yhzhu99/pyehr project
    data/ # import user uploaded data, merge data tables, stats...
    pipeline/ # model training and evaluation, ...
    analysis/ # analysis modules
    plot/ # plot modules
```

## Sample Usages

### Pipeline of Training and Predicting

```python
import pandas as pd

from xiaoya.data import DataHandler
from xiaoya.pipeline import Pipeline

labtest_data = pd.read_csv('datasets/raw_labtest_data.csv')
events_data = pd.read_csv('datasets/raw_events_data.csv')
target_data = pd.read_csv('datasets/raw_target_data.csv')
data_handler = DataHandler(labtest_data=labtest_data, events_data=events_data, target_data=target_data)
data_handler.execute()

pl = Pipeline()
result = pl.execute()
print(result)
```

### Analysis and Plot

* Dataset Visualization

```python
import pandas as pd

from xiaoya.data import DataHandler
from xiaoya.plot import plot_vis_dataset

labtest_data = pd.read_csv('datasets/raw_labtest_data.csv')
events_data = pd.read_csv('datasets/raw_events_data.csv')
target_data = pd.read_csv('datasets/raw_target_data.csv')
data_handler = DataHandler(labtest_data=labtest_data, events_data=events_data, target_data=target_data)
data_handler.execute()
result = data_handler.analyze_dataset()
plot_vis_dataset(result['detail'], save_path='./output/vis_data')
```

* Plot Feature Importance histogram

```python
import pandas as pd

from xiaoya.pipeline import Pipeline
from xiaoya.analysis import DataAnalyzer
from xiaoya.plot import plot_feature_importance

pl = Pipeline(model='ConCare')
pl.execute()

data_analyzer = DataAnalyzer(pl.config, pl.model_path)
train_raw = pd.read_csv('datasets/train_raw.csv')
train_x = pd.read_pickle('datasets/train_x.pkl')
result = data_analyzer.feature_importance(
    df=train_raw,
    x=train_x,
    patient_index=0
)
plot_feature_importance(result, save_path='./output/')
```

* Plot Patient Risk curve

```python
import pandas as pd

from xiaoya.pipeline import Pipeline
from xiaoya.analysis import DataAnalyzer
from xiaoya.plot import plot_risk_curve

pl = Pipeline(model='ConCare')
pl.execute()

data_analyzer = DataAnalyzer(pl.config, pl.model_path)
train_raw = pd.read_csv('datasets/train_raw.csv')
train_x = pd.read_pickle('datasets/train_x.pkl')
train_mask = pd.read_pickle('datasets/train_missing_mask.pkl')
train_mean = pd.read_pickle('datasets/train_mean.pkl')
train_std = pd.read_pickle('datasets/train_std.pkl')
result, time, time_risk = data_analyzer.risk_curve(
    df=train_raw,
    x=train_x,
    mean=train_mean,
    std=train_std,
    mask=train_mask,
    patient_index=0
)
plot_risk_curve(result, time, time_risk, save_path='./output/')
```

* Plot Patient Embedding and Trajectory

```python
import pandas as pd

from xiaoya.pipeline import Pipeline
from xiaoya.analysis import DataAnalyzer
from xiaoya.plot import plot_patient_embedding

pl = Pipeline(model='ConCare')
pl.execute()

data_analyzer = DataAnalyzer(pl.config, pl.model_path)
train_raw = pd.read_csv('datasets/train_raw.csv')
train_x = pd.read_pickle('datasets/train_x.pkl')
train_mean_age = pd.read_pickle('datasets/train_mean.pkl')['Age']
train_std_age = pd.read_pickle('datasets/train_std.pkl')['Age']
result = data_analyzer.data_dimension_reduction(
    df=train_raw,
    x=train_x,
    mean_age=train_mean_age,
    std_age=train_std_age
)
plot_patient_embedding(result, save_path='./output/')
```

* AI Advice

```python
import pandas as pd

from xiaoya.pipeline import Pipeline
from xiaoya.analysis import DataAnalyzer

pl = Pipeline(model='ConCare')
pl.execute()

data_analyzer = DataAnalyzer(pl.config, pl.model_path)
train_raw = pd.read_csv('datasets/train_raw.csv')
train_x = pd.read_pickle('datasets/train_x.pkl')
train_mean = pd.read_pickle('datasets/train_mean.pkl')
train_std = pd.read_pickle('datasets/train_std.pkl')
result = data_analyzer.ai_advice(
    df=train_raw,
    x=train_x,
    mask=train_mask,
    patient_index=0,
    time_index=-1
)
print(result)
```
