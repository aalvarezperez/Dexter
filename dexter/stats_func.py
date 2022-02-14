from numpy import round, sqrt
from scipy.stats import chisquare, t, norm


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


def mde(xn, yn, yvar, xvar, alpha=.05, beta=1 - .8, alternative='two-sided'):
    assert alternative in ['two-sided', 'one-sided']

    dsd = sqrt(xvar + yvar)

    alpha = alpha / 2 if alternative == 'two-sided' else alpha

    dof = yn + xn - 2

    t_critical = t.ppf(1 - alpha, df=dof)
    t_beta = t.ppf(beta, df=dof)

    return t_critical * dsd + t_beta * dsd


def required_n(xmean, ymean, yvar, xvar, alpha=.05, beta=1 - .8, alternative='two-sided'):
    assert alternative in ['two-sided', 'one-sided']

    dsd = sqrt(xvar + yvar)

    alpha = alpha / 2 if alternative == 'two-sided' else alpha

    delta = xmean - ymean

    t_critical = norm.ppf(1 - alpha)
    t_beta = norm.ppf(beta)

    return (t_critical + t_beta) ** 2 * dsd / delta ** 2
