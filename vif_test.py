import pandas as pd
import numpy as np
import seaborn as sns
from patsy import dmatrices
import statsmodels.api as sm;
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import StandardScaler

df = pd.read_csv('fred_data.csv')

y_column = "FEDFUNDS"

x_columns = [name for name in df.columns if name not in [y_column, "year", "month"]]

# original model
new_df = df.copy()
scaler1 = StandardScaler()
scaler2 = StandardScaler()
scaler3 = StandardScaler()

lm = sm.OLS(new_df[y_column], scaler1.fit_transform(new_df[x_columns]))
results = lm.fit()
print("original model")
print(results.summary())
print()

y, X = dmatrices(f'{y_column} ~ ' + " + ".join(x_columns), df, return_type = 'dataframe')

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
print(vif)
print()
new_x_columns = vif[vif["VIF Factor"]<50].features.to_list()

lm = sm.OLS(df[y_column], scaler2.fit_transform(new_df[new_x_columns]))
results = lm.fit()
print("new model 1")
print(results.summary())
print()

new_x_columns = vif[vif["VIF Factor"]>=50].features.to_list()
new_x_columns.remove("Intercept")

lm = sm.OLS(df[y_column], scaler3.fit_transform(new_df[new_x_columns]))
results = lm.fit()
print("new model 2")
print(results.summary())
print()