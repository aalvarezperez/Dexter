import pytest
import pandas as pd
from numpy import log
from dexter.experiment import ExperimentDataFrame
from dexter.analyser import ExperimentAnalyser

df = pd.read_csv('/Users/aalvarezperez/Documents/eBay/Horizon/projects/Dexter/dummy_df.csv')

exp_df = ExperimentDataFrame(
    dataframe=df.copy(),
    success_metric='leads',
    health_metrics='revenue',
    learning_metrics='vips',
    experiment_unit='userid',
    treatment='treatment',
    expected_proportions=[.5, .5]
    )


class TestTransformMetrics(object):
    def test_log_transform(self):
        analyser = ExperimentAnalyser(exp_df)
        analyser.transform_metrics(['leads'], log)
        actual = analyser._data['leads'][0]
        expected = log(df['leads'])[0]
        assert actual == pytest.approx(expected)