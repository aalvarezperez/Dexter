from typing import Any
from stats_func import trim_outliers, winsorize_outliers, check_multiple_proportion
from numpy import round, mean, sum, ndarray
from utils import indent, print_nested_dict


def print_status_message(status_dict):
    print_nested_dict(status_dict)


class ExperimentChecker:
    def __init__(self):

        self._crossover_mask = None

        self._log = {
            'group_balance': {
                'assumption': 'the group sizes have the pre-defined proportions',
                'info': 'if proportions deviate from the pre-defined ones, issues may exist in the randomisation.',
                'status': {
                    'checked': False,
                    'passed': bool,
                    'handled': False,
                    'diagnostics': {
                        'groups': list,
                        'observed prop': list,
                        'expected prop': list,
                        'differences': list,
                        'test results': {'test': 'chi-square GOF', 'statistic': float, 'p-value': float}
                    },
                }
            },
            'crossover': {
                'assumption': 'experiment units are exposed to a single variant only (no cross-over)',
                'info': 'if an experiment unit is exposed to multiple variants, '
                        'then the average treatment effect may be biased.',
                'status': {
                    'checked': False,
                    'passed': bool,
                    'handled': False,
                    'diagnostics': {
                        'cross-over cases': int,
                        'percent of total': str
                    }
                }
            },
            'outliers': {
                'assumption': 'depends on the definition of outliers that is used.',
                'outliers': 'carefully handling outliers will likely result in increased statistical power.',
                'status': {
                    'checked': False,
                    'passed': bool,
                    'handled': False,
                    'diagnostics': {
                        'method': None,
                        'affected metrics': list,
                        'number of affected units': int,
                        'stats': object
                    }
                 }
            }
        }

    def get_log(self):
        return self._log

    def check_groups_balance(self, data, expected_proportion):
        df = data.df
        treatment = data.treatment

        if len(expected_proportion) != df[treatment].nunique():
            raise ValueError('Provide a proportion for each group available in the treatment column.')

        n_treatment = df[treatment].value_counts().sort_index()

        n_total = df.shape[0]

        observed_prop = round(n_treatment / n_total, 3).tolist()

        test_res = check_multiple_proportion(n_total, n_treatment, expected_proportion)

        differences: list[ndarray | Any] = [round(e - o, 3) for e, o in zip(expected_proportion, observed_prop)]

        self._log['group_balance']['status']['checked'] = True
        self._log['group_balance']['status']['passed'] = test_res[1] > .05
        self._log['group_balance']['status']['diagnostics']['groups'] = data.groups
        self._log['group_balance']['status']['diagnostics']['observed prop'] = observed_prop
        self._log['group_balance']['status']['diagnostics']['expected prop'] = expected_proportion
        self._log['group_balance']['status']['diagnostics']['differences'] = differences
        self._log['group_balance']['status']['diagnostics']['test results']['statistic'] = test_res[0]
        self._log['group_balance']['status']['diagnostics']['test results']['p-value'] = test_res[1]

        print_status_message(self._log.get('group_balance'))

    def check_crossover(self, data):
        df = data.df
        experiment_unit = data.experiment_unit
        treatment = data.treatment

        user_treatment_df = df.drop_duplicates(subset=[experiment_unit, treatment])
        user_freq_series = user_treatment_df[experiment_unit].value_counts()
        crossed_over = user_freq_series > 1

        absolute = crossed_over.sum()
        percent = '{}%'.format(round(crossed_over.mean() * 100))

        self._crossover_mask = crossed_over
        self._log['crossover']['status']['checked'] = True
        self._log['crossover']['status']['passed'] = absolute == 0
        self._log['crossover']['status']['diagnostics']['cross-over cases'] = absolute
        self._log['crossover']['status']['diagnostics']['percent of total'] = percent

        print_status_message(self._log.get('crossover'))

    def check_outliers(self, data, is_outlier, metrics, func):

        func = [func] if not isinstance(func, list) else func

        df = data.df

        aggr_df = df[metrics].groupby(is_outlier).agg(func)

        aggr_df.rename(index={True: 'Outliers', False: 'Regulars'}, inplace=True)

        aggr_df.index.name = 'Stratum'

        aggr_df = aggr_df.round(3)

        deltas = ((aggr_df.loc['Outliers', :] / aggr_df.loc['Regulars', :]) - 1) * 100

        aggr_df.loc['(delta)'] = [f'({x:.3g}%)' for x in deltas]

        self._log['outliers']['status']['diagnostics']['stats'] = aggr_df

        if self._crossover_mask is None:
            self._crossover_mask = is_outlier

        print_status_message(self._log.get('outliers'))

        # because print_status_message will not print a dataframe, as aggr_df is
        print('Stats:\n', aggr_df, '\n')

    def handle_crossover(self):

        print('• Handling cross-overs...')

        if self._crossover_mask is None:
            print(indent('Nothing to take care of. Have you ran the check for this assumption first?'))
            print('')
            return
        if sum(self._crossover_mask == 0):
            print(indent('There are no cross-over cases to handle. You are good to go.'))
            print('')
            return

        crossover_thresh = .01

        if mean(self._crossover_mask) > crossover_thresh:
            raise ValueError(f'More than {round(crossover_thresh * 100, 3)}% units were exposed to multiple variants',
                             'It is not recommended to proceed without better understanding why this is the case.')

        self._log['crossover']['status']['handled'] = True

        delta_rows = len(self._crossover_mask)

        print(f'{delta_rows} units were removed from the working dataset.')

    def handle_outliers(self, data, method, is_outlier=None, metrics=None):

        print('• Handling outliers...')

        if is_outlier is None:
            if self._crossover_mask is not None:
                is_outlier = self._crossover_mask
            else:
                raise Exception('is_outlier can not be None if you have not provided it earlier, while checking.')
        else:
            print('Overriding definition of is_outlier...')
            print('')
            self._crossover_mask = is_outlier

        if metrics is None:
            print(indent('All success and learning metrics are affected by default. See the "metrics" argument.'))
            metrics = [*data.success_metric, *data.learning_metrics]

        method_dict = {
            'trim': trim_outliers,
            'winsorize': winsorize_outliers
        }

        outlier_fun = method_dict[method]

        data.df = outlier_fun(dataframe=data.df, outlier_mask=is_outlier, metrics=metrics)

        total_affected = is_outlier.sum()

        percent_affected = is_outlier.mean()

        self._log['outliers']['status']['handled'] = True
        self._log['outliers']['status']['diagnostics']['method'] = method
        self._log['outliers']['status']['diagnostics']['affected metrics'] = metrics
        self._log['outliers']['status']['diagnostics']['number of affected units'] = total_affected

        print(indent('{} experiment units were affected: {}% of the total sample.\n'
              .format(total_affected, round(percent_affected*100, 3))))
