import numpy as np
import pandas as pd
from sklearn import linear_model
from word2number import w2n
import math

df = pd.read_csv('hiring.csv')
#print(df)
df.experience = df.experience.fillna("zero")
df.experience = df.experience.apply(w2n.word_to_num)
mts = math.floor(df['test_score(out of 10)'].mean())
#print(mts)
df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(mts)
reg = linear_model.LinearRegression()
reg.fit(df.drop('salary($)', axis='columns'), df['salary($)'])
#linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
print(reg.predict([[2, 9, 6]]))
print(reg.predict([[12, 10, 10]]))
