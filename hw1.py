import pandas as pd 
import numpy as np

df = pd.read_csv('mlzoomcamp_hw1.csv')

#Question 1:
print(pd.__version__)

#Question 2:
print(len(df.columns))

#Question 3:
print(df.isnull().sum())

#Question 4:
print(df.ocean_proximity.nunique())

#Question 5:
print(df.groupby('ocean_proximity').median_house_value.mean())

#Question 6:
total_bedrooms_mean = df.total_bedrooms.mean()
print(total_bedrooms_mean)

df.total_bedrooms.fillna(total_bedrooms_mean)
print(df.total_bedrooms.mean())

#Question 7:
df = df.loc[df['ocean_proximity'] == 'ISLAND']
df = df[['housing_median_age', 'total_rooms', 'total_bedrooms']]
x = df.values
xtx = np.dot(x.T, x)
xtx_inv = np.linalg.inv(xtx)
y = [950, 1300, 800, 1000, 1300]

w = xtx_inv.dot(x.T).dot(y)
print(w)