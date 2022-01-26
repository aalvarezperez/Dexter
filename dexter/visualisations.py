import seaborn as sns
import pandas as pd
from itertools import chain, repeat


class ResultsVisualiser:
    def __init__(self, log):
        self._log = log

    def _plot_group_balance(self):

        log = self._log['group_balance']

        observed = log['status']['diagnostics']['observed prop']
        expected = log['status']['diagnostics']['expected prop']
        groups = log['status']['diagnostics']['groups']

        assert len(observed) == len(expected) == len(groups)

        df = pd.DataFrame({
            'Variant': groups.tolist() * len(expected),
            'Frequency': observed + expected,
            'Kind': list(chain.from_iterable(zip(*repeat(['Observed', 'Expected'], len(observed)))))
        })

        sns.catplot(y='Frequency', x='Variant', hue='Kind', kind='bar', data=df)

    def _plot_outliers(self):

        log = self._log['outliers']

        df = log['status']['diagnostics']['stats'].reset_index()

        df_melt = df.melt(id_vars='Stratum')

        df_melt.rename({'variable_0': 'Metric', 'variable_1': 'Statistic', 'value': 'Value'}, inplace=True, axis=1)

        df_melt = df_melt[df_melt.Stratum.isin(['Regulars', 'Outliers'])]

        sns.catplot(data=df_melt, x='Statistic', y='Value', hue='Stratum', col='Metric', kind='bar')

    def plot_assumption(self, assumption):

        assumption_dict = {
            'group_balance': self._plot_group_balance,
            'outliers': self._plot_outliers
        }

        assumption_plot_fun = assumption_dict[assumption]

        return assumption_plot_fun()
