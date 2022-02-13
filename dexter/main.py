from experiment import Experiment, ExperimentDataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/aalvarezperez/Documents/eBay/Horizon/projects/Dexter/dummy_df.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)
df.rename({'treatment': 'group'}, axis=1, inplace=True)  # rename, as 'treatment' is a forbidden column name.

exp_df = ExperimentDataFrame(
    dataframe=df,
    success_metric='leads',
    health_metric='revenue',
    learning_metrics='vips',
    experiment_unit='userid',
    treatment='group',
    expected_proportions=[.5, .5]
    )

exp = Experiment(
    experiment_name='great_exp',
    start='2021-01-01',
    end='2021-01-14',
    expected_delta=.3,
    roll_out_percent=.1
    )

################################
### read out experiment ########
################################

exp.read_out(exp_df)

# exp.describe_data(by='revenue')

################################
### check assumptions ##########
################################

exp.assumptions.check_groups_balance()

exp.assumptions.check_crossover()

exp.assumptions.check_outliers(
    metrics=['vips', 'leads'],
    is_outlier=exp.data.leads > 1,
    func=[np.mean, np.median]
    )

################################
###  fix assumptions  ##########
################################

print(exp.mde[0].mde)

# exp.assumptions.handle_crossover()

# exp.assumptions.handle_outliers(method='trim', is_outlier=exp.data.leads > 1, metrics=['leads', 'vips'])
#
#
# ################################
# ###  analyze experiment  #######
# ################################
#
# exp.analyser.transform_metrics_log(['leads', 'vips'], offset=1)
#
#
# exp.analyser.compare(parametric=False, func=None)

# # exp.visualiser.plot_conditional(y='revenue', x='leads', group='group')
# exp.visualiser.plot_assumption('outliers')
#
# plt.show()

