import pandas as pd 
import numpy as np

df = pd.read_csv('/Users/chosen/mlzoomcamp/mlzoomcamp_hw1.csv')

# Data prepration
df = df[df['ocean_proximity'].isin(['<1H OCEAN', 'INLAND'])]
df = df[['latitude', 'longitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value']]

#Question 1
print(df.isnull().sum())
#Question 2
print(df['population'].median())

#Data prepration for training/validating/testing
n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test
idx = np.arange(n)
np.random.seed(42)
np.random.shuffle(idx)
df_train = df.iloc[idx[:n_train]].reset_index(drop=True)
df_val = df.iloc[idx[n_train:n_train+n_val]].reset_index(drop=True)
df_test = df.iloc[idx[n_train+n_val:]].reset_index(drop=True)

y_train = np.log1p(df_train.median_house_value.values)
y_val = np.log1p(df_val.median_house_value.values)
y_test = np.log1p(df_test.median_house_value.values)


#Question 3 
def train_linear_regression(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    return w_full[0], w_full[1:]


def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)


def prepare_X(df, method:int):
    df_num = df
    df_num = df_num.fillna(method)
    X = df_num.values
    return X

#fill_na
X_train = prepare_X(df_train, 0)
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val, 0)
y_pred = w0 + X_val.dot(w)
fill_na_result = round(rmse(y_val, y_pred), 2)

#mean
mean = df_train['median_house_value'].mean()
X_train = prepare_X(df_train, mean)

w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val, mean)
y_pred = w0 + X_val.dot(w)
fill_mean_result = round(rmse(y_val, y_pred), 2)

print(fill_na_result)
print(fill_mean_result)

# #Question 4
def train_linear_regression_reg(X, y, r:float):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)
    
    return w_full[0], w_full[1:]


r_list = [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]
for r in r_list:
    X_train = prepare_X(df_train, 0)
    w0, w = train_linear_regression_reg(X_train, y_train, r)

    X_val = prepare_X(df_val, 0)
    y_pred = w0 + X_val.dot(w)
    reg_result = round(rmse(y_val, y_pred), 2)
    print(f"{r} result : {reg_result}")

#Question 5

n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = n - n_val - n_test
idx = np.arange(n)
seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
result_list = []
for i in seed_list:
    np.random.seed(i)
    np.random.shuffle(idx)
    df_train = df.iloc[idx[:n_train]].reset_index(drop=True)
    df_val = df.iloc[idx[n_train:n_train+n_val]].reset_index(drop=True)
    df_test = df.iloc[idx[n_train+n_val:]].reset_index(drop=True)

    y_train = np.log1p(df_train.median_house_value.values)
    y_val = np.log1p(df_val.median_house_value.values)
    y_test = np.log1p(df_test.median_house_value.values)

    X_train = prepare_X(df_train, 0)
    w0, w = train_linear_regression(X_train, y_train)

    X_val = prepare_X(df_val, 0)
    y_pred = w0 + X_val.dot(w)
    fill_na_result = round(rmse(y_val, y_pred), 2)
    result_list.append(fill_na_result)


print(result_list)
print(round(np.std(result_list), 3))

#Question 6
n = len(df)
n_test = int(n * 0.2)
n_train = n - n_test
idx = np.arange(n)
np.random.seed(9)
np.random.shuffle(idx)
df_train = df.iloc[idx[:n_train]].reset_index(drop=True)
df_test = df.iloc[idx[n_train+n_test:]].reset_index(drop=True)

y_train = np.log1p(df_train.median_house_value.values)
y_test = np.log1p(df_test.median_house_value.values)

X_train = prepare_X(df_train, 0)
w0, w = train_linear_regression_reg(X_train, y_train, 0.001)

X_val = prepare_X(df_test, 0)
y_pred = w0 + X_val.dot(w)
reg_result = round(rmse(y_test, y_pred), 2)
print(reg_result)



