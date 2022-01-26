import pandas as pd

df = pd.DataFrame.from_dict({('vips', 'mean'): {'Regulars': 5.1584150667614015, 'Outliers': 4.984258050193952}, ('leads', 'mean'): {'Regulars': 0.4552362291119115, 'Outliers': 3.0022929965973932}})

df = df.round(3)

df.loc['(delta)'] = [f'({x:.3g}%)' for x in ((df.loc['Outliers',:] / df.loc['Regulars',:]) - 1)*100]