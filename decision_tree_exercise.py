import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split


df = pd.read_csv('titanic.csv')
#print(df)
df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns', inplace=True)

inputs = df.drop('Survived', axis='columns')
#print(inputs)
target = df['Survived']

inputs.Sex = inputs.Sex.map({'male': 1, 'female': 2})
inputs.Age = inputs.Age.fillna(inputs.Age.mean())

#le_pclass = LabelEncoder()
#le_sex = LabelEncoder()
#le_age = LabelEncoder()
#le_fare = LabelEncoder()

#inputs['pclass_n'] = le_pclass.fit_transform(inputs['Pclass'])
#inputs['sex_n'] = le_sex.fit_transform(inputs['Sex'])
#inputs['age_n'] = le_age.fit_transform(inputs['Age'])
#inputs['fare_n'] = le_fare.fit_transform(inputs['Fare'])

#print(inputs)
#inputs_n = inputs.drop(['Pclass', 'Sex', 'Age', 'Fare'], axis='columns')

X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2)
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
