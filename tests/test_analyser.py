import pandas as pd
import pytest
from numpy import log
import dexter as dex

# from dexter.analyser import ExperimentAnalyser
from dexter.experiment import ExperimentDataFrame
# from dexter.utils import *

df = pd.read_csv('/Users/aalvarezperez/Documents/eBay/Horizon/projects/Dexter/dummy_df.csv')


# exp_df = ExperimentDataFrame(
#     dataframe=df,
#     success_metric='leads',
#     health_metrics='revenue',
#     learning_metrics='vips',
#     experiment_unit='userid',
#     treatment='treatment',
#     expected_proportions=[.5, .5]
# )
#
#
# class TestTransformMetrics(object):
#     def test_log_transform(self):
#         analyser = ExperimentAnalyser(exp_df)
#         analyser.transform_metrics(['leads'], log)
#         actual = analyser.self._data['leads']
#         expected = log(exp_df['leads'])
#         assert actual == pytest.approx(expected)
