from sklearn.cluster import k_means
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

df = pd.read_csv('income.csv')
#print(df)

print(plt.scatter(df.Age, df['Income($)']))
print(plt.xlabel('Age'))
print(plt.ylabel('Income($)'))


