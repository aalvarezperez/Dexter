from Experiment import Experiment, ExperimentDataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('/Users/aalvarezperez/Documents/eBay/Horizon/projects/Dexter/dummy_df.csv')

exp_df = ExperimentDataFrame(
    data=df,
    success_metric='leads',
    health_metrics='revenue',
    learning_metrics='vips',
    experiment_unit='userid',
    treatment='treatment'
)


exp = Experiment(
    experiment_name='great_exp',
    start='2021-01-01',
    end='2021-01-14',
    expected_delta=.3,
    treatment_proportions=[.5, .5],
    roll_out_percent=.1
)

exp.read_out(exp_df)

exp.check_group_balance()

exp.check_outliers(metrics=['vips', 'leads'], is_outlier=exp.data.leads > 1, func=[np.mean, np.median, np.std])

exp.handle_outliers(exp.data.leads > 1, 'winsorize')

exp.handle_crossover()

exp.print_assumption_checks()

exp.transform_metrics(['leads', 'vips'], np.log)

exp.calculate_lift(parametric=True, func=None)

exp.visualiser.plot_assumption('outliers')

plt.show()


