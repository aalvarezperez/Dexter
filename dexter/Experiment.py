from assumptions import ExperimentChecker
from analysis import ExperimentAnalyser
from visualisations import ResultsVisualiser
from numpy import sort, random
from utils import *


class ExperimentDataFrame:
    def __init__(
            self,
            data,
            success_metric,
            health_metrics,
            learning_metrics,
            treatment,
            group_proportions,
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
        self.group_proportions = group_proportions
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
    
    def __init__(self, experiment_name, start, end, expected_delta, roll_out_percent):
        """
        This method creates a new experiment obejct.
        """
        
        self._data = None
        self.experiment_name = experiment_name
        self.start, self.end = start, end
        self.expected_delta = expected_delta
        self.roll_out_percent = roll_out_percent

        self.assumptions = ExperimentChecker(data=self._data)
        self.analyses = ExperimentAnalyser(data=self._data)
        self.visualiser = ResultsVisualiser(log=self.assumptions.get_log())

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
            self.assumptions = ExperimentChecker(data=self._data)
            self.analyses = ExperimentAnalyser(data=self._data)
            self.visualiser = ResultsVisualiser(log=self.assumptions.get_log())
        else:
            raise ValueError('Input must be of type ExperimentDataFrame.')


        

               
            

        
        
    
    

    
    

    
    

