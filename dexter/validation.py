import warnings
import pandas
from numpy import sort

from dexter.utils import strcol
from abc import ABC, abstractmethod

warnings.formatwarning = lambda msg, *args, **kwargs: f'{msg}\n'


class _BaseValidator(ABC):

    def __set_name__(self, owner, name):
        self.private_name = '_' + name

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name)

    def __set__(self, obj, value):
        self.validate(value)
        setattr(obj, self.private_name, value)

    @abstractmethod
    def validate(self, value):
        pass


class _String(_BaseValidator):
    def validate(self, value):
        if not isinstance(value, str):
            raise ValueError('Input should be a string.')


class _DataFrame(_BaseValidator):
    def validate(self, value):
        if not isinstance(value, pandas.DataFrame):
            raise ValueError(f'dataframe should be of type Pandas DataFrame, '
                             f'but got {type(value).__name__} instead.')
        if value.shape[0] == 0:
            raise ValueError('experiment_df is empty')


class _ExperimentDataFrame(_BaseValidator):
    def validate(self, value):
        if value.__class__.__name__ != 'ExperimentDataFrame' and value is not None:
            raise ValueError(f'dataframe should be of type Dexter ExperimentDataFrame, '
                             f'but got {value.__class__.__name__} instead.')


class _ColumnIdentifier(_String):
    def __init__(self, forbidden):
        self.forbidden = forbidden

    def validate(self, value):
        # _String.validate(self, value)
        if value in self.forbidden:
            raise ValueError('The column identifier cannot be the same as any of the reserved attribute names for the '
                             'ExperimentDataFrame class. Change the column name in the original pandas DataFrame.')


class _Metric(_ColumnIdentifier):
    def __init__(self, forbidden):
        _ColumnIdentifier.__init__(self, forbidden)

    def __set__(self, obj, value):
        self.validate(value)
        value = [value] if not isinstance(value, list) else value
        setattr(obj, self.private_name, value)

    def validate(self, value):
        _ColumnIdentifier.validate(self, value)


class _TestMetric(_Metric):
    def validate(self, value):
        _Metric.validate(self, value)
        if len(value) > 2:
            warnings.warn(f'It is not recommended to have more than 1 or 2 test metrics (i.e., success and health '
                          f'metrics). '
                          f'Consider excluding some or moving them to learning metrics')


class _ExpectedProportions(_BaseValidator):
    def validate(self, value):
        if sum(value) != 1:
            raise ValueError('The provided proportions should sum up to 1, exactly, '
                             'and a proportion is required for each group.')


class _Number(_BaseValidator):
    def validate(self, value):
        if not isinstance(value, (float, int)):
            raise ValueError('Input should be either a float or an int.')


class _Proportion(_Number):
    def validate(self, value):
        _Number.validate(self, value)
        if value < 0 or value > 1:
            raise ValueError('Input should be a proportion: a number between, and including, 0 and 1.')


def _post_validate_experiment_dataframe(obj):
    groups = sort(obj.data[obj.treatment].unique())

    if obj.data.shape[0] > 2 * 10 ** 6:
        print(strcol('Info: it is recommended to delete the original DataFrame after initialising it as '
                     'an ExperimentDataFrame, to save working memory', 'warning'))

    groups_threshold = 7
    if len(groups) > groups_threshold:
        warnings.warn(f'More than {groups_threshold} were detected. Are you sure you want to proceed?\n'
                      f'Info: the more groups the more tests and, therefore, '
                      f'the higher the chance to find an effect mistakenly')

    if len(obj.expected_proportions) != len(groups):
        raise ValueError('The number of expected proportions provided does not match'
                         'the number of groups in the treatment column.')

    if obj.data[obj.experiment_unit].nunique() < obj.data.shape[0]:
        warnings.warn('There seems to be repeating experiment units. This causes a problem for most statistical '
                      'analyses. Consider investigating the cause for this. If reasonable, you can handle this case'
                      'with the Experiment.handle_crossover() method.')
