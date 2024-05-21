import  pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('HR_comma_sep.csv')
#print(df.head())

#Data Expolaration and Visualisation
left = df[df.left == 1]
#print(left.shape)

retained = df[df.left == 0]
#print(retained.shape)

#Average No of Columns
#print(df.groupby('left').mean())

#print(df.groupby('left')[['satisfaction_level','last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company','Work_accident', 'promotion_last_5years' ]].mean())

#From above table we can draw following conclusions,

#**Satisfaction Level**: Satisfaction level seems to be relatively low (0.44) in employees leaving the firm vs the retained ones (0.66)
#**Average Monthly Hours**: Average monthly hours are higher in employees leaving the firm (199 vs 207)
#**Promotion Last 5 Years**: Employees who are given promotion are likely to be retained at firm

subdf = df[['satisfaction_level', 'average_montly_hours', 'promotion_last_5years', 'salary']]
#print(subdf.head())

sal_dummy = pd.get_dummies(subdf.salary, prefix='salary')
df_dummy = pd.concat([subdf, sal_dummy], axis='columns')
#print(df_dummy)
df_dummy = df_dummy.drop('salary', axis='columns')
#print(df_dummy.head())

X = df_dummy
y = df.left

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3)
model = LogisticRegression()
model.fit(X_train, y_train)

print(model.predict(X_test))
print(model.score(X_test, y_test))

