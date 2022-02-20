import functools
import builtins
import itertools

from tabulate import tabulate
from pandas.core.frame import DataFrame
import numpy as np

def strcol(string, modification=None):
    if modification is None:
        return string

    modification_dict = {
        'header': '\033[95m',
        'okblue': '\033[94m',
        'okcyan': '\033[96m',
        'okgreen': '\033[92m',
        'warning': '\033[93m',
        'fail': '\033[91m',
        'endc': '\033[0m',
        'bold': '\033[1m',
        'underline': '\033[4m'
        }

    mod = modification_dict[modification]
    end = modification_dict['endc']

    return mod + string + end


def pretty_results(df, floatfmt=".3f", tablefmt='simple', title=None, subtitle=None, note=None):
    """Pretty display of results table.
    Parameters
    ----------
    df : :py:class:`pandas.DataFrame`
        Dataframe to print (e.g. ANOVA summary)
    floatfmt : string
        Decimal number formatting
    tablefmt : string
        Table format (e.g. 'simple', 'plain', 'html', 'latex', 'grid', 'rst').
        For a full list of available formats, please refer to
        https://pypi.org/project/tabulate/
    """
    assert type(title) == str or title is None
    assert type(subtitle) == str or subtitle is None
    assert type(note) == str or note is None

    if title is not None:
        title = title.replace('_', ' ').capitalize()

        print(f'\n==========\n{title}\n==========\n')

    if subtitle is not None:
        print(indent(subtitle))

    print(indent(tabulate(df, headers="keys", showindex=True, floatfmt=floatfmt, tablefmt=tablefmt), 2))
    print('')

    if note is not None:
        print(indent(note, 2))


def indent(txt, indents=1):
    assert txt is not None
    return '\n'.join(' ' * 4 * indents + ln for ln in txt.splitlines())


def print_nested_dict(dict_obj, indent=0, exclude_keys=[]):
    """Pretty Print nested dictionary with given indent level"""

    exclude_keys = [exclude_keys] if not isinstance(exclude_keys, list) else exclude_keys

    for key, value in dict_obj.items():

        if key in exclude_keys or isinstance(value, DataFrame):
            continue

        key = key.capitalize().replace('_', ' ')

        if isinstance(value, dict):
            print(' ' * indent, key)
            print_nested_dict(value, indent + 4, exclude_keys=exclude_keys)
        else:
            print(' ' * indent, key, ':', value)


def _customise_res_table(res):
    col_name_dict = {
        'Source': 'Source',
        'A': 'A',
        'B': 'B',
        'mean(A)': 'mean(A)',
        'mean(B)': 'mean(B)',
        'diff': 'delta',
        'SS': 'SS',
        'DF': 'dof',
        'dof': 'dof',
        'ddof1': 'dof',
        'F': 'f-stat',
        'se': 'stderr',
        'H': 'H-stat',
        'T': 't-stat',
        'U': 'u-stat',
        'p-unc': 'p-value',
        'p-tukey': 'p-value',
        'p-corr': 'p-value (adj)',
        'cohen': 'effect size (d)'
        }

    res = res.copy()

    selected_cols = res.columns.isin(col_name_dict.keys())

    res = res.loc[:, selected_cols]

    res.rename(
        {old: new for (old, new) in col_name_dict.items()},
        axis=1,
        inplace=True
        )

    return res


def default_metrics(experiment):
    return [*experiment.data.success_metric, *experiment.data.health_metrics]


def function_details(func):
    """Add signature to return statement of a function"""

    argnames = func.__code__.co_varnames[:func.__code__.co_argcount]
    fname = func.__name__
    @functools.wraps(func)
    def wrapper_signature_name(*args, **kwargs):
        base_args_repr = ['%s=%r' % entry for entry in zip(argnames, args[:len(argnames)])]
        args_repr = [f'arg{args.index(a)}={a}' for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(base_args_repr[1:] + kwargs_repr)
        value = func(*args, **kwargs)
        return value, signature, fname

    return wrapper_signature_name


def pinfo(*values, color: str = None, do_print=True, **kwargs):
    values = ['Info: ' + v for v in values]
    values = tuple(strcol(value, modification=color) for value in values)
    if not do_print:
        return values
    builtins.print(*values, **kwargs)


def expand_grid(data_dict):
    rows = itertools.product(*data_dict.values())
    return DataFrame.from_records(rows, columns=data_dict.keys())


def prep_actual_power(data: DataFrame, treatment_col: str, metrics: list) -> DataFrame:
    stats_df = data.groupby(treatment_col)[metrics].agg(
        {
            'mean',
            'var',
            'count'
            }
        ) \
        .rename(columns={'count': 'n'}) \
        .transpose() \
        .reset_index()

    stats_df = stats_df.rename(columns={'level_0': 'metric', 'level_1': 'aggr'})

    df_melt = stats_df.melt(id_vars=['metric', 'aggr']).set_index(['metric', 'aggr']).reset_index()
    df_melt = df_melt.pivot_table(index=['metric', 'group'], columns=['aggr'])
    df_melt = df_melt.droplevel(0, 1)
    df_melt['group'] = df_melt.index.get_level_values('group')
    df_melt = df_melt.droplevel('group')
    df_joined = df_melt.add_prefix('x').join(df_melt.add_prefix('y'))
    df_joined = df_joined.loc[df_joined.xgroup != df_joined.ygroup]

    df_joined = df_joined.set_index([df_joined.index, 'xgroup', 'ygroup']) \
        .rename(index={'xgroup': 'A', 'ygroup': 'B'})
    df_joined['delta'] = df_joined['xmean'] - df_joined['ymean']
    df_joined['abs_delta'] = np.abs(df_joined['delta'])
    df_final = df_joined.drop_duplicates(subset=['abs_delta']).drop(columns=['abs_delta'])

    return df_final
