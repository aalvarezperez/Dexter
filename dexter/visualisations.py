import seaborn as sns
import pandas as pd
from numpy import mean
from itertools import chain, repeat


class ExperimentVisualiser:
    def __init__(self, experiment):
        self.experiment = experiment

    def _plot_group_balance(self):
        source = self.experiment.assumptions.get_log()
        source = source['group_balance']

        observed = source['diagnostics']['observed prop']
        expected = source['diagnostics']['expected prop']

        groups = source['diagnostics']['groups']

        assert len(observed) == len(expected) == len(groups)

        df = pd.DataFrame({
            'Variant': groups.tolist() * len(expected),
            'Frequency': observed + expected,
            'Kind': list(chain.from_iterable(zip(*repeat(['Observed', 'Expected'], len(observed)))))
            })

        sns.catplot(y='Frequency', x='Variant', hue='Kind', kind='bar', data=df) \
            .set(title='Expected vs. observed proportions of the experiment groups.')

    def _plot_outliers(self):
        source = self.experiment.assumptions.get_log()
        source = source['outliers']

        df = pd.DataFrame.from_dict(source['diagnostics']['stats'])

        df_melt = df.melt(ignore_index=False)

        df_melt.rename({'variable_0': 'Metric', 'variable_1': 'Statistic', 'value': 'Value'}, inplace=True, axis=1)

        df_melt = df_melt[df_melt.index.isin(['Regulars', 'Outliers'])]

        df_melt['Stratum'] = df_melt.index

        sns.catplot(data=df_melt, x='Statistic', y='Value', hue='Stratum', col='Metric', kind='bar')

    def plot_conditional(self, y, x, group):
        source = self.experiment.data

        bins = pd.qcut(source[x], q=10)

        res = source.groupby([group, bins])[y].mean().reset_index()

        res[x] = res[x].astype(str)

        sns.lineplot(y=res[y], x=res[x], hue=res[group])

    def plot_assumption(self, assumption):
        source = self.experiment.assumptions.get_log()
        assumptions = source.keys()

        if assumption not in assumptions:
            raise AttributeError(f'assumption should be one of: {", ".join([s for s in assumptions])}')

        assumption_dict = {
            'group_balance': self._plot_group_balance,
            'outliers': self._plot_outliers
            }

        assumption_plot_fun = assumption_dict[assumption]

        return assumption_plot_fun()
