from numpy import round, sqrt
from scipy.stats import chisquare, t


def trim_outliers(dataframe, outlier_mask, metrics=None):
    return dataframe.loc[~outlier_mask]


def winsorize_outliers(dataframe, outlier_mask, metrics):
    metrics = [metrics] if type(metrics) is not list else metrics
    max_val = dataframe.loc[outlier_mask, metrics].min().values
    dataframe.loc[outlier_mask, metrics] = max_val
    return dataframe


def check_multiple_proportion(n_total, n_treatment, expected_proportion):
    observed_prop = n_treatment / n_total
    res = chisquare(expected_proportion, observed_prop)
    return round(res, 3)


def mde_binomial(x, y, alpha=.05, beta=1 - .8, alternative='two-sided', xn=None, yn=None):
    assert alternative in ['two-sided', 'one-sided']

    ymean = y.mean()
    xmean = x.mean()
    yn = y.shape[0] if yn is None else yn
    xn = x.shape[0] if xn is None else xn

    yvar = ymean * (1 - ymean) / yn
    xvar = xmean * (1 - xmean) / xn
    dsd = sqrt(xvar + yvar)

    alpha = alpha / 2 if alternative == 'two-sided' else alpha

    dof = yn + xn - 1

    t_critical = t.ppf(1 - alpha, df=dof)
    t_beta = t.ppf(1 - beta, df=dof)

    return abs(-t_critical * dsd - t_beta * dsd)


def mde_continuous(x, y, alpha=.05, beta=1 - .8, alternative='two-sided', xn=None, yn=None, yvar=None, xvar=None):
    assert alternative in ['two-sided', 'one-sided']

    yn = y.shape[0] if yn is None else yn
    xn = x.shape[0] if xn is None else xn

    yvar = y.var() if yvar is None else yvar
    xvar = x.var() if xvar is None else xvar
    dsd = sqrt(xvar + yvar)

    alpha = alpha / 2 if alternative == 'two-sided' else alpha

    dof = yn + xn - 1

    t_critical = t.ppf(1 - alpha, df=dof)
    t_beta = t.ppf(1 - beta, df=dof)

    return abs(-t_critical * dsd - t_beta * dsd)
