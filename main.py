import pandas as pd

# read csv and save it as pandas dataframe
interest_rate_data = pd.read_excel('fred_data.xlsx')

interest_rate_data["FEDFUNDS_DIFF"] = interest_rate_data["FEDFUNDS"].diff(1)
interest_rate_data = interest_rate_data.fillna(0)

# print data shape
print("data shape: ", interest_rate_data.shape)

# print column names
print(interest_rate_data.columns)

col = interest_rate_data.columns.to_list()
col.remove('year')
col.remove('month')
col.remove('FEDFUNDS')
col.remove("FEDFUNDS_DIFF")


from itertools import combinations
import math

# print(math.comb(24,3))
# print(list(combinations(col,3)))
# print(len(list(combinations(col,3))))
# print(list(combinations(col,3))[0])
# print(list(combinations(col,3))[0][0])

# machine learning
import sklearn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import tqdm

def linModel(X_list):
  # You need to choose variables
  # X_list = [*list(combinations(col,3))[i]]
  # print(X_list)

  # Split-out train and test dataset
  # start from Jan 1990
  # train samples: years 1990 to 2010, test samples: years 2011 to 2023
  interest_rate_data_test = interest_rate_data.loc[(interest_rate_data['year'] >= 2011)].copy()
  interest_rate_data_train = interest_rate_data.loc[(interest_rate_data['year'] >= 1990) & (interest_rate_data['year'] <= 2010)].copy()

  X_train = (interest_rate_data_train[X_list]).values
  y_train = (interest_rate_data_train['FEDFUNDS']).values
  X_test = (interest_rate_data_test[X_list]).values
  y_test = (interest_rate_data_test['FEDFUNDS']).values

  # Make predictions on validation dataset
  model = LinearRegression()
  #model = Ridge(alpha=1)
  #model = Lasso(alpha=0.05)

  # Standardization: necessary for Lasso or Ridge regression, does not affect linear regression
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  # fit and predict
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  # Evaluate predictions
  # The coefficients
  # print('Coefficients:')
  # for i in range(len(X_list)):
    # print(X_list[i],": %.3f" % model.coef_[i])

  # print()

  # The mean squared error, correlation, r-square
  # print('ML - Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
  # print('ML - Correlation: %.2f' % np.corrcoef(y_test, y_pred)[0,1])
  # print('ML - R square: %.2f' % r2_score(y_test, y_pred))
  # print()

  # equilibrium real fed funds rate (assumed to be 2.5%)
  # desired (target) inflation rate (assumed to be 2%)
  #taylor = inflation+2.5+0.5*(inflation-2)+0.5*output_gap
  interest_rate_data['Taylor'] = interest_rate_data['INFLATION']+2.5+0.5*(interest_rate_data['INFLATION']-2)+0.5*interest_rate_data['OUTPUT_GAP']
  y_taylor_pred = interest_rate_data['Taylor'].loc[(interest_rate_data['year'] >= 2011)].values

  # The mean squared error, correlation, r-square
  # print('Taylor - Mean squared error: %.2f' % mean_squared_error(y_test, y_taylor_pred))
  # print('Taylor - Correlation: %.2f' % np.corrcoef(y_test, y_taylor_pred)[0,1])
  # print('Taylor - R square: %.2f' % r2_score(y_test, y_taylor_pred))

  result = [X_list,
            mean_squared_error(y_test, y_pred),np.corrcoef(y_test, y_pred)[0,1],
            r2_score(y_test, y_pred), mean_squared_error(y_test, y_taylor_pred),
            np.corrcoef(y_test, y_taylor_pred)[0,1],r2_score(y_test, y_taylor_pred)]

  return result


combinations_num = list(combinations(col, 2))

result_dict = {
    "index": [],
    "ML-MSE": [],
    "ML-Corr": [],
    "ML-R2": [],
}
for list_of_x in tqdm.tqdm(combinations_num):
    index, ml_mse, ml_corr, ml_r2, t_mse, t_corr, t_r2 = linModel(list(list_of_x))
    result_dict["index"].append(index)
    result_dict["ML-MSE"].append(ml_mse)
    result_dict["ML-Corr"].append(ml_corr)
    result_dict["ML-R2"].append(ml_r2)

result_df = pd.DataFrame(result_dict)
result_df.to_csv("./result_x_2.csv")
