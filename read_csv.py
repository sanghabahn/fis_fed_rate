import pandas as pd

df = pd.read_csv(r'/Users/sanghabahn/PycharmProjects/FIS_FED_FUNDS/result_x_5.csv')

result_dict={
    "index":[],
    "ML-MSE":[],
    "ML-Corr":[],
    "ML-R2":[],
}
result_df = pd.DataFrame(df)
new_df = result_df.sort_values("ML-R2", ascending=False)

print(new_df.head())

for i in range(0,5):
    print(new_df.values[i])
    print("")