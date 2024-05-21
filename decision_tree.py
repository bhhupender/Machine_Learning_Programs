import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('salaries.csv')
#print(df)

inputs = df.drop('salary_more_then_100k', axis='columns')
target = df['salary_more_then_100k']

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_company.fit_transform(inputs['job'])
inputs['degree_n'] = le_company.fit_transform(inputs['degree'])

#print(inputs)

inputs_n = inputs.drop(['company','job','degree'], axis='columns')
#print(inputs_n)

model = tree.DecisionTreeClassifier()
model.fit(inputs_n, target)

#Is salary of Google, Computer Engineer, Bachelors degree > 100 k ?
print(model.predict([[2, 1, 0]]))

#Is salary of Google, Computer Engineer, Masters degree > 100 k ?
print(model.predict([[2, 1, 1]]))
