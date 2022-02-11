import pandas
from numpy import sort

from dexter.analyser import ExperimentAnalyser
from dexter.assumptions import ExperimentChecker
from dexter.utils import *
import dexter.validation as validation
from dexter.visualisations import ExperimentVisualiser


class ExperimentDataFrame:
    _forbidden = ['treatment', 'groups', 'n_groups', 'success_metric', 'learning_metrics',
                  'health_metric', 'experiment_unit', 'expected_proportions', 'dataframe']

    success_metric = validation._TestMetric(_forbidden)
    health_metrics = validation._TestMetric(_forbidden)
    learning_metrics = validation._Metric(_forbidden)
    experiment_unit = validation._ColumnIdentifier(_forbidden)
    treatment = validation._ColumnIdentifier(_forbidden)
    expected_proportions = validation._ExpectedProportions()
    data = validation._DataFrame()

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
        validation._post_validate_experiment_dataframe(self)

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
    experiment_name = validation._String()
    start = validation._String()
    end = validation._String()
    roll_out_percent = validation._Proportion()
    expected_delta = validation._Number()
    _data = validation._ExperimentDataFrame()

    def __init__(
            self,
            experiment_name: str,
            start: str,
            end: str,
            expected_delta: float,
            roll_out_percent: float,
            experiment_df: ExperimentDataFrame = None
            ):
        """
        This method creates a new experiment object.
        """

        self.experiment_name = experiment_name
        self.start = start
        self.end = end
        self.expected_delta = expected_delta
        self.roll_out_percent = roll_out_percent
        self._data = experiment_df

        if self._data is not None:
            self.assumptions = ExperimentChecker(self)
            self.analyser = ExperimentAnalyser(self)
            self.visualiser = ExperimentVisualiser(self)
        else:
            pinfo('you initialised the experiment, but there is no data to analyse yet. '
                  'See the .read_out() method.', color='warning')

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value: ExperimentDataFrame):
        if not isinstance(value, ExperimentDataFrame):
            raise ValueError('Data should be of Dexter ExperimentDataFrame type.')
        self._data = value

    @property
    def groups(self):
        data = self._data
        return sort(data[data.treatment].unique())

    @property
    def n_groups(self):
        return len(self.groups)

    @property
    def sample_size(self):
        return self.data.shape[0]

    def read_out(self, data: ExperimentDataFrame):
        self._data = data
        self.assumptions = ExperimentChecker(self)
        self.analyser = ExperimentAnalyser(self)
        self.visualiser = ExperimentVisualiser(self)
        pinfo('experiment dataframe has been read.', color='okgreen')

    def describe_data(self, by: str = None, q: int = 3):

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
