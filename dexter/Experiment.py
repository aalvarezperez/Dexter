from assumptions import ExperimentChecker
from analysis import ExperimentAnalyser
from visualisations import ResultsVisualiser
from numpy import sort, random, mean
from utils import *
from textwrap import dedent



class ExperimentDataFrame:
    def __init__(
            self,
            data,
            success_metric,
            health_metrics,
            learning_metrics,
            treatment,
            experiment_unit
    ):
        if not isinstance(data, DataFrame):
            raise ValueError('Data should be a pandas DataFrame')

        success_metric = [success_metric] if type(success_metric) is not list else success_metric
        health_metrics = [health_metrics] if type(health_metrics) is not list else success_metric
        learning_metrics = [learning_metrics] if type(learning_metrics) is not list else success_metric

        self.df = data
        self.groups = sort(data[treatment].unique())
        self.n_groups = data[treatment].nunique()
        self.treatment = treatment
        self.experiment_unit = experiment_unit
        self.success_metric = success_metric
        self.health_metrics = health_metrics
        self.learning_metrics = learning_metrics


class Experiment:
    """
    The Experiment class initiates an experiment object, which gives the user access to methods that represent the 
    several steps of analysing an A/B test:
    
    1. Reading out
    2. Checking assumptions
    3. Aliviating possible violations
    4. Calculating lift
    5. Visualising results
    """
    
    def __init__(self, experiment_name, start, end, expected_delta, treatment_proportions, roll_out_percent):
        """
        This method creates a new experiment obejct.
        """
        
        self._data = None
        self.experiment_name = experiment_name
        self.start, self.end = start, end
        self.expected_delta = expected_delta
        self.treatment_proportions = treatment_proportions
        self.roll_out_percent = roll_out_percent

        self.assumptions = ExperimentChecker()
        self.analyses = ExperimentAnalyser()
        self.visualiser = ResultsVisualiser(self.assumptions.get_log())

    @property
    def data(self):
        return self._data.df

    @data.setter
    def data(self, new_data):
        if isinstance(new_data, ExperimentDataFrame):
            self._data = new_data
        else:
            ValueError('Input must be of ExperimentDataFrame type.')

    @data.deleter
    def data(self):
        raise AttributeError('You cannot delete the data of an experiment. You can start over by reading out new data')

    def read_out(self, experiment_df):
        if isinstance(experiment_df, ExperimentDataFrame):
            self._data = experiment_df
            self.assumptions = ExperimentChecker()
            self.analyses = ExperimentAnalyser()
            self.visualiser = ResultsVisualiser(self.assumptions.get_log())
        else:
            raise ValueError('Input must be of type ExperimentDataFrame.')

    def check_group_balance(self):
        self.assumptions.check_groups_balance(self._data, self.treatment_proportions)

    def check_crossover(self):
        self.assumptions.check_crossover(self._data)

    def check_outliers(self, is_outlier, metrics, func=mean):
        self.assumptions.check_outliers(self._data, is_outlier=is_outlier, metrics=metrics, func=func)

    def check_all_assumptions(self):
        self.check_group_balance()
        self.check_crossover()

    def handle_crossover(self):
        if self.assumptions.get_log()['crossover']['status']['checked'] is False:
            self.check_crossover()
        self.assumptions.handle_crossover()

    def handle_outliers(self, is_outlier, method, metrics=None):
        self.assumptions.handle_outliers(data=self._data, method=method, is_outlier=is_outlier, metrics=metrics)

    def print_assumption_checks(self):
        header_str = """
        =================================================================================
        Assumptions check status
        =================================================================================
        """
        print(dedent(header_str))
        print_nested_dict(self.assumptions.get_log())

    def transform_metrics(self, metrics, transform_func):
        self.analyses._transform_metrics(data=self._data.df, metrics=metrics, transform_func=transform_func)

    def transform_metrics_log(self, metrics, offset=0):
        self.analyses._transform_metrics_log(data=self.data, metrics=metrics, offset=offset)

    def calculate_lift(
            self, metrics=None, alternative='two-sided', alpha=.05, paired=False, parametric=True,
            padjust='none', func=None, rounds=1000, method='approx', seed=random.randint(1, 10000)
                       ):

        self.analyses._calculate_lift(
            data=self._data,
            alpha=alpha,
            padjust=padjust,
            metrics=metrics,
            parametric=parametric,
            alternative=alternative,
            paired=paired,
            func=func,
            rounds=rounds,
            method=method,
            seed=seed
        )


        

               
            

        
        
    
    

    
    

    
    

