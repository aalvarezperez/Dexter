from pandas import qcut
from dexter.assumptions import ExperimentChecker
from dexter.analyser import ExperimentAnalyser, BaseAnalyser
from dexter.visualisations import ResultsVisualiser
from numpy import sort
from dexter.utils import *


class ExperimentDataFrame:
    def __init__(
            self,
            success_metric,
            health_metrics,
            learning_metrics,
            experiment_unit,
            treatment,
            expected_proportions,
            dataframe=None
            ):

        success_metric = [success_metric] if type(success_metric) is not list else success_metric
        health_metrics = [health_metrics] if type(health_metrics) is not list else success_metric
        learning_metrics = [learning_metrics] if type(learning_metrics) is not list else success_metric

        self._df = dataframe
        self.success_metric = success_metric
        self.health_metrics = health_metrics
        self.learning_metrics = learning_metrics
        self.experiment_unit = experiment_unit
        self.treatment = treatment
        self.expected_proportions = expected_proportions
        self.groups = sort(dataframe[treatment].unique())
        self.n_groups = dataframe[treatment].nunique()

        if self._df.shape[0] > 2*10**6:
            print(strcol('Info: it is recommended to delete the original DataFrame after initialising it as '
                         'an ExperimentDataFrame, to save working memory', 'warning'))

    @property
    def data(self):
        return self._df

    @data.setter
    def data(self, df):
        if not isinstance(df, DataFrame):
            raise AttributeError('You can only assign a pandas DataFrame to this ExperimentDataFrame.data.')
        self._df = df

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._df, attr)

    def __getitem__(self, item):
        return self._df[item]

    def __setitem__(self, item, data):
        self._df[item] = data



class Experiment:
    """
    The Experiment class initiates an experiment object, which gives the user access to methods that represent the 
    several steps of analysing an A/B tests:
    
    1. Reading out
    2. Checking assumptions
    3. Aliviating possible violations
    4. Calculating lift
    5. Visualising results
    """
    
    def __init__(self, experiment_name, start, end, expected_delta, roll_out_percent, data=None):
        """
        This method creates a new experiment obejct.
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
                df['stratum'] = qcut(df[by], q, precision=3)
                df.drop([by], inplace=True, axis=1)

            df.rename({by: 'stratum'}, axis=1, inplace=True)

            strata = sort(df['stratum'].reset_index(drop=True).unique())

            for stratum in strata:
                title = f'{by}: {stratum}'
                frame = '\n' + '='*(len(title)+1) + '\n'
                print(
                    frame,
                    title,
                    frame
                    )
                mask = df['stratum'] == stratum
                pretty_results(self._data.loc[mask].describe())
        else:
            pretty_results(self._data.describe())


