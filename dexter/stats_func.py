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


def mde(xn, yn, yvar, xvar, alpha, beta, alternative):
    assert alternative in ['two-sided', 'one-sided']

    alpha = alpha / 2 if alternative == 'two-sided' else alpha

    dof = yn + xn - 2

    t_critical = t.ppf(1 - alpha, df=dof)
    t_beta = t.ppf(1 - beta, df=dof)

    mde_squared = (t_critical + t_beta) ** 2 * ((xvar * 1 / xn) + (yvar * 1 / yn))

    return sqrt(mde_squared)


def required_n(xmean, ymean, xvar, yvar, alpha, beta, alternative):
    assert alternative in ['two-sided', 'one-sided']

    alpha = alpha / 2 if alternative == 'two-sided' else alpha

    delta = xmean - ymean
    t_critical = norm.ppf(1 - alpha)
    t_beta = norm.ppf(1 - beta)

    return (yvar + xvar) * (t_critical + t_beta) ** 2 / (delta ** 2)


def actual_power(xmean, ymean, xvar, yvar, xn, yn, alpha, alternative):
    assert alternative in ['two-sided', 'one-sided']

    alpha = alpha / 2 if alternative == 'two-sided' else alpha

    delta = xmean - ymean
    dsd = sqrt(xvar * 1 / xn + yvar * 1 / yn)
    dof = xn + yn - 2
    t_critical = t.ppf(1 - alpha, df=dof)
    t_beta = norm.ppf(t_critical - delta/dsd)
    print(delta, dsd, t_critical, dof, t_critical - delta/dsd)

    return 1 - t_beta
