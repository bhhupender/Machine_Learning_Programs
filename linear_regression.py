import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

df = pd.read_csv('canada_per_capita_income.csv')
#print(df)
new_df = df.drop('per_capita_income',axis='columns')
#print(new_df)
income = df.per_capita_income
#print(income)
# Create linear regression object
reg = linear_model.LinearRegression()
reg.fit(new_df,income)
print(reg.predict([[2020]]))