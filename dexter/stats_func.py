from scipy.stats import chisquare
from numpy import round

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
