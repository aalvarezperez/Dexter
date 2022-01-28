import pingouin as pg
import numpy as np
import pandas as pd
import random
from utils import pretty_results, strcol, _customise_res_table

class ExperimentAnalyser:

    def __init__(self, data):
        self._data = data
        self._log = {
            'transformations': {},
            'analyses': {}
        }

    def get_log(self, part=None):
        assert part in ['transformations', 'analyses', None]
        return self._log[part] if part is not None else self._log

    def transform_metrics(self, metrics, func):

        df = self._data.df

        if not callable(func):
            raise ValueError('transform_func has to be a callable that takes a single argument.')

        for metric in metrics:
            df.loc[:, metric] = func(df[metric])
            self._log['transformations'][f'{metric}'] = str(func)

        print(f'Info: the following metrics were transformed with {func.__name__}: {", ".join(x for x in metrics)}.')

    def transform_metrics_log(self, metrics, offset=0):
        assert offset >= 0
        def log(x):
            return np.log(x + offset)
        self.transform_metrics(metrics, func=log)

    def compare(self,
                alpha=.05,
                padjust='none',
                alternative='two-sided',
                paired=False,
                metrics=None,
                parametric=True,
                func=None,
                rounds=1000,
                method='approx',
                seed=random.randint(1, 10000)
                ):

        data = self._data.df
        metrics = [*data.success_metric, *data.learning_metrics] if metrics is None else metrics
        treatment = data.treatment
        n_groups = data.n_groups
        groups = data.groups
        df = data.df

        if parametric == 'permute':
            if n_groups > 2:
                raise Exception('Permutations are not enabled for experiment with more than two variants.')

            calculator = PermutationComparison(
                data=df,
                metrics=metrics,
                treatment=treatment,
                alpha=alpha,
                padjust=padjust,
                parametric=parametric,
                alternative=alternative,
                paired=paired,
                groups=groups,
                func=func,
                method=method,
                rounds=rounds,
                seed=seed
            )

        elif n_groups > 2:

            calculator = MultipleComparison(
                data=df,
                metrics=metrics,
                treatment=treatment,
                alpha=alpha,
                padjust=padjust,
                parametric=parametric,
                alternative=alternative,
                paired=paired,
                groups=groups
            )

        elif n_groups == 2:

            calculator = SingleComparison(
                data=df,
                metrics=metrics,
                treatment=treatment,
                alpha=alpha,
                parametric=parametric,
                padjust=padjust,
                alternative=alternative,
                paired=paired
            )

        calculator.run()

        self._log['analyses'] = calculator.results



class BaseAnalyser:

    def __init__(self, data, metrics, treatment, alternative, padjust, parametric, paired, alpha, groups=None):

        if alternative not in ('two-sided', 'greater', 'smaller'):
            raise AttributeError(f'method should be either two-sided, greater or smaller. Got {alternative} instead.')

        if not isinstance(paired, bool):
            raise AttributeError('paired should be boolean.')

        if not isinstance(parametric, bool) and parametric != 'permute':
            raise AttributeError('paired should be boolean or "permute".')

        if alpha < 0 or alpha > 1:
            raise AttributeError('alpha should be a proportion.')

        self.data = data
        self.metrics = metrics
        self.treatment = treatment
        self.groups = groups
        self.alpha = alpha
        self.alternative = alternative
        self.padjust = padjust
        self.parametric = parametric
        self.paired = paired
        self.equal_var_dict = None
        self.results = {}
        # TODO effect size should be set according to metric type: continuous/binary

    def _check_homoskedasticity(self):

        equal_var_dict = {}

        for metric in self.metrics:
            check_homoskedasticity = pg.homoscedasticity(
                data=self.data,
                dv=metric,
                group=self.treatment,
                method='levene'
            )

            equal_var = check_homoskedasticity.loc['levene', 'equal_var']

            equal_var_dict[metric] = equal_var

        self.equal_var_dict = equal_var_dict


class SingleComparison(BaseAnalyser):

    def _run_ttest(self, metric, equal_var):

        res = pg.pairwise_ttests(
            data=self.data,
            dv=metric,
            between=self.treatment,
            padjust=self.padjust,
            parametric=self.parametric,
            effsize='cohen',  # TODO effect size should be set according to metric type: continuous/binary
            correction=equal_var if equal_var else 'auto'
        )

        res = _customise_res_table(res)

        note = f'Info: Welch\'s test is applied automatically if metric variance across the experiment variants differs.'

        pretty_results(res, title=metric, subtitle='T-test:', note=note)

        if metric not in self.results:
            self.results[metric] = {}

        self.results[metric]['t-test'] = res.to_dict()


    def run(self):

        self._check_homoskedasticity()

        for metric, equal_var in self.equal_var_dict.items():

            self._run_ttest(metric, equal_var)



class MultipleComparison(SingleComparison):

    def _run_anova(self, metric, equal_var):

        # outer dict: parametric
        # inner dict: equal variance
        method = {
            True: {
                True: pg.anova,
                False: pg.welch_anova
            },
            False: {
                True: pg.kruskal,
                False: pg.kruskal
            }
        }

        anova = method[self.parametric][equal_var]

        res = anova(
            data=self.data,
            dv=metric,
            between=self.treatment,
            detailed=True,
        )

        res = _customise_res_table(res)

        if metric not in self.results:
            self.results[metric] = {}

        self.results[metric]['anova'] = res.to_dict()

        if res.loc[0, 'p-value'] <= self.alpha:
            note = f'Info: the treatment has an effect on {metric}. ' \
                         'It is warranted to examine the contrasts between groups in post-hoc.'

            note = strcol(note, 'okgreen')

        elif res.loc[0, 'p-value'] > self.alpha:
            note = f'Info: (none of) the treatment(s) has any effect on {metric}. ' \
                         'Relying on the post-hoc test may lead to false positive findings (type-I error)'

            note = strcol(note, 'fail')
        else:
            note = None

        pretty_results(res, title=metric, subtitle='ANOVA:', note=note)



    def _run_posthoc(self, metric, equal_var):

        # True if variances across groups are equal
        method = {
            True: pg.pairwise_tukey,
            False: pg.pairwise_gameshowell
        }

        posthoc = method[equal_var]

        res = posthoc(
            data=self.data,
            dv=metric,
            between=self.treatment,
            effsize='cohen'  # TODO effect size should be set according to metric type: continuous/binary
        )

        res = _customise_res_table(res)

        if metric not in self.results:
            self.results[metric] = {}

        self.results[metric]['post_hoc'] = res.to_dict()

        test = 'Tukey\'s test' if equal_var else 'Games-Howell'

        note = 'p-values are adjusted for multiple analyses (see Tukey\'s and Games-Howell test)'

        pretty_results(res, title=None, subtitle=f'Post-hoc ({test}):', note=note)



    def run(self):

        self._check_homoskedasticity()

        for metric, equal_var in self.equal_var_dict.items():

            self._run_anova(metric, equal_var)

            self._run_posthoc(metric, equal_var)



class PermutationComparison(BaseAnalyser):
    def __init__(
            self, data, metrics, treatment, alternative, padjust, parametric, paired, alpha, groups,
            func, method, rounds, seed
    ):
        BaseAnalyser.__init__(self, data, metrics, treatment, alternative, padjust, parametric, paired, alpha, groups)

        if func is None:
            print('Permuting for mean difference by default. Use a custom function in arg func for median and quantiles.')
            if alternative == 'two-sided':
                def func(a, b):
                    a_stat = a.mean()
                    b_stat = b.mean()
                    diff = abs(a_stat - b_stat)
                    return (a_stat, b_stat, diff)
                self.func = func
            elif alternative == 'greater':
                def func(a, b):
                    a_stat = a.mean()
                    b_stat = b.mean()
                    diff = a_stat - b_stat
                    return (a_stat, b_stat, diff)

                self.func = func
            elif alternative == 'smaller':
                def func(a, b):
                    a_stat = a.mean()
                    b_stat = b.mean()
                    diff = abs(b_stat - a_stat)
                    return (a_stat, b_stat, diff)

                self.func = func

        self.func = func
        self.method = method
        self.rounds = rounds
        self.seed = seed

    def _unpaired_perm(self, metric):

        a, b = [self.data.loc[self.data[self.treatment] == g, metric] for g in self.groups]

        rng = np.random.RandomState(self.seed)

        k = len(a)

        null_dist = np.hstack([a, b])

        a_stat, b_stat, observed_delta = self.func(a, b)

        total = past_observed = 0

        permutations = []

        for i in range(self.rounds):

            rng.shuffle(null_dist)

            perm_delta = self.func(null_dist[:k], null_dist[k:])[2]

            if perm_delta >= observed_delta:
                past_observed += 1

            total += 1

        p = past_observed / total

        results = pd.DataFrame({
            'A': [0],
            'B': [1],
            'stat(A)': [a_stat],
            'stat(B)': [b_stat],
            'diff': [observed_delta],
            'permutations': [self.rounds],
            'p-value': [p]
        })

        if metric not in self.results:
            self.results[metric] = {}

        self.results[metric]['permutation-test'] = results

        pretty_results(results, title=metric, subtitle='Permutation test')

    def run(self):

        if self.paired:
            print('paired permutations not implemented yet')

        elif not self.paired:

            for metric in self.metrics:

                self._unpaired_perm(metric)



