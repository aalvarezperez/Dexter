import pandas
from numpy import sort

from dexter.analyser import ExperimentAnalyser
from dexter.assumptions import ExperimentChecker
from dexter.utils import *
from dexter.validation import _TestMetric, _Metric, _ColumnIdentifier, _DataFrame, \
    _ExpectedProportions, _post_validate_experiment_dataframe
from dexter.visualisations import ResultsVisualiser


class ExperimentDataFrame:
    forbidden = ['treatment', 'groups', 'n_groups', 'success_metric', 'learning_metrics',
                 'health_metric', 'experiment_unit', 'expected_proportions', 'dataframe']

    success_metric = _TestMetric(forbidden)
    health_metrics = _TestMetric(forbidden)
    learning_metrics = _Metric(forbidden)
    experiment_unit = _ColumnIdentifier(forbidden)
    treatment = _ColumnIdentifier(forbidden)
    expected_proportions = _ExpectedProportions()
    data = _DataFrame()

    def __init__(
            self,
            success_metric: list[str],
            health_metric: list[str],
            learning_metrics: list[str],
            experiment_unit: str,
            treatment: str,
            expected_proportions: list[float],
            dataframe: pandas.DataFrame
            ):
        self.data = dataframe
        self.success_metric = success_metric
        self.health_metrics = health_metric
        self.learning_metrics = learning_metrics
        self.experiment_unit = experiment_unit
        self.treatment = treatment
        self.expected_proportions = expected_proportions
        self._post_validate()

    def _post_validate(self):
        _post_validate_experiment_dataframe(self)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.data, attr)

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, item, data):
        self.data[item] = data


class Experiment:
    """
    The Experiment class initiates an experiment object, which gives the user access to methods that represent the 
    several steps of analysing an A/B tests:
    
    1. Reading out
    2. Checking assumptions
    3. Alleviating possible violations
    4. Calculating lift
    5. Visualising results
    """

    def __init__(self, experiment_name, start, end, expected_delta, roll_out_percent, data=None):
        """
        This method creates a new experiment object.
        """

        self.experiment_name = experiment_name
        self.start, self.end = start, end
        self.expected_delta = expected_delta
        self.roll_out_percent = roll_out_percent

        self._data = data
        self.assumptions = ExperimentChecker(data=self._data)
        self.analyser = ExperimentAnalyser(data=self._data)
        self.visualiser = ResultsVisualiser(source={
            'assumptions': self.assumptions.get_log(),
            'analyses': self.analyser.get_log(),
            'data': self._data
            })

        if self._data is None:
            print(strcol('Info: you initialised the experiment, but there is no data to analyse yet.'
                         ' See the .read_out() method.', 'warning')
                  )

    @property
    def groups(self):
        return sort(self.data[self.data.treatment].unique())

    @property
    def n_groups(self):
        return len(self.groups)

    @property
    def sample_size(self):
        return self.data.shape[0]

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data):
        if isinstance(new_data, ExperimentDataFrame):
            self._data = new_data
        else:
            ValueError('Input must be of ExperimentDataFrame type.')

    def read_out(self, data):
        if isinstance(data, ExperimentDataFrame):
            self._data = data
            self.assumptions = ExperimentChecker(data=self._data)
            self.analyser = ExperimentAnalyser(data=self._data)
            self.visualiser = ResultsVisualiser(source={
                'assumptions': self.assumptions.get_log(),
                'analyses': self.analyser.get_log(),
                'data': self._data
                })
            print(strcol('Info: experiment dataframe has been read.', 'okgreen'))
        else:
            raise ValueError('Input must be of type ExperimentDataFrame.')

    def describe_data(self, by=None, q=3):

        df = self._data.data.copy()

        high_cardinality = df[by].nunique() > 7

        if by is not None:
            if high_cardinality:
                df['stratum'] = pandas.qcut(df[by], q, precision=3)
                df.drop([by], inplace=True, axis=1)

            df.rename({by: 'stratum'}, axis=1, inplace=True)

            strata = sort(df['stratum'].reset_index(drop=True).unique())

            for stratum in strata:
                title = f'{by}: {stratum}'
                frame = '\n' + '=' * (len(title) + 1) + '\n'
                print(
                    frame,
                    title,
                    frame
                    )
                mask = df['stratum'] == stratum
                pretty_results(self._data.loc[mask].describe())
        else:
            pretty_results(self._data.describe())
