from collections import namedtuple

import pandas
from numpy import sort, mean

import dexter.validation as validation
from dexter.analyser import ExperimentAnalyser
from dexter.assumptions import ExperimentChecker
from dexter.stats_func import mde, required_n, actual_power
from dexter.utils import *
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
    data = validation._ExperimentDataFrame()

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
        self.data = experiment_df

        if self.data is not None:
            self.assumptions = ExperimentChecker(self)
            self.analyser = ExperimentAnalyser(self)
            self.visualiser = ExperimentVisualiser(self)
        else:
            pinfo('you initialised the experiment, but there is no data to analyse yet. '
                  'See the .read_out() method.', color='warning')

    @property
    def groups(self):
        data = self.data
        return sort(data[data.treatment].unique())

    @property
    def n_groups(self):
        return len(self.groups)

    @property
    def sample_size(self):
        return self.data.shape[0]

    def mde(self, metrics=None, alpha=.05, beta=1 - .8, alternative='two-sided'):
        """
        Minimum detectable effect given observed sample sizes, variances, and provided type I and type II levels.

        Post-hoc power analysis is considered a bad practice.
        [reference] explain that the power of a test provides no more information once the p-value is known.
        Therefore, knowing the p-value of a test and still doing a post-hoc power is a circular.
        And so, post-hoc power analysis may reinforce the mistaken belief that the obtained p-value adheres to the
        overall set levels of type I error.

        :return:
        minimum detectable effect: list[tuple(metric, mde)]
        """

        data = self.data
        treatment = data.treatment
        metrics = default_metrics(self) + data.learning_metrics if metrics is None else metrics
        metrics = [metrics] if not isinstance(metrics, list) else metrics

        _tmp = data.value_counts(treatment).sort_index()
        control_idx = 0
        smallest_n_testgroup_idx = _tmp[1:].argmin() + 1

        count_df = _tmp.iloc[[control_idx, smallest_n_testgroup_idx]].rename('var')

        stats_df = data.groupby(treatment)[metrics].agg(['var']).transpose()

        stats_df.columns = ['xvar', 'yvar']

        stats_df[['xn', 'yn']] = count_df

        arguments = stats_df.to_dict(orient='records')

        min_det_effect = namedtuple('mde', ['metric', 'mde'])

        results = []
        for metric, kwargs in zip(metrics, arguments):
            metric_value_pair = min_det_effect(metric, mde(**kwargs, alpha=alpha, beta=beta, alternative=alternative))
            results.append(metric_value_pair)

        return results

    def required_n(self, metrics=None, alpha=.05, beta=1 - .8, alternative='two-sided'):
        """
        Minimum detectable effect given observed sample sizes, variances, and provided type I and type II levels.

        Post-hoc power analysis is considered a bad practice.
        [reference] explain that the power of a test provides no more information once the p-value is known.
        Therefore, knowing the p-value of a test and still doing a post-hoc power is a circular.
        And so, post-hoc power analysis may reinforce the mistaken belief that the obtained p-value adheres to the
        overall set levels of type I error.

        :return:
        minimum detectable effect: list[tuple(metric, mde)]
        """

        data = self.data
        treatment = data.treatment
        metrics = default_metrics(self) + data.learning_metrics if metrics is None else metrics
        metrics = [metrics] if not isinstance(metrics, list) else metrics

        stats_df = data.groupby(treatment)[metrics].agg([mean, 'var']).transpose()

        stats_df = stats_df.unstack()

        stats_df.columns = ['xmean', 'xvar', 'ymean', 'yvar']

        arguments = stats_df.to_dict(orient='records')

        sample_size = namedtuple('sample_size', ['metric', 'n'])

        results = []

        for metric, kwargs in zip(metrics, arguments):
            metric_value_pair = sample_size(
                metric,
                np.ceil(required_n(**kwargs, alpha=alpha, beta=beta, alternative=alternative))
                )
            results.append(metric_value_pair)

        return results

    def actual_power(self, metrics=None, alpha=.05, alternative='two-sided'):
        """
        Minimum detectable effect given observed sample sizes, variances, and provided type I and type II levels.

        Post-hoc power analysis is considered a bad practice.
        [reference] explain that the power of a test provides no more information once the p-value is known.
        Therefore, knowing the p-value of a test and still doing a post-hoc power is a circular.
        And so, post-hoc power analysis may reinforce the mistaken belief that the obtained p-value adheres to the
        overall set levels of type I error.

        :return:
        minimum detectable effect: list[tuple(metric, mde)]
        """

        data = self.data
        treatment = data.treatment
        metrics = default_metrics(self) + data.learning_metrics if metrics is None else metrics
        metrics = [metrics] if not isinstance(metrics, list) else metrics

        arguments_df = prep_actual_power(data=data, treatment_col=treatment, metrics=metrics)

        arguments_df['power'] = arguments_df.apply(
            lambda row:
            actual_power(
                row.xmean, row.ymean,
                row.xvar, row.yvar,
                row.xn, row.yn,
                alpha=alpha,
                alternative=alternative
                ),
            axis=1
            )

        return arguments_df


    def read_out(self, data: ExperimentDataFrame):
        self.data = data
        self.assumptions = ExperimentChecker(self)
        self.analyser = ExperimentAnalyser(self)
        self.visualiser = ExperimentVisualiser(self)
        pinfo('experiment dataframe has been read.', color='okgreen')

    #  todo: re-write describe_data() so that it does not make a copy of the original data
    def describe_data(self, by: str = None, q: int = 3):

        df = self.data.data.copy()

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
                pretty_results(self.data.loc[mask].describe())
        else:
            pretty_results(self.data.describe())
