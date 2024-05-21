import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

df = pd.read_csv('titanic.csv')
#print(df)

df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns', inplace=True)
#print(df)

inputs = df.drop('Survived', axis='columns')
target = df['Survived']

# inputs.Sex = inputs.Sex.map({'male': 1, 'female': 2})
dummies = pd.get_dummies(inputs.Sex)
inputs = pd.concat([inputs, dummies], axis='columns')
inputs.drop(['Sex', 'male'], axis='columns',inplace=True)
#print(inputs)

#print(inputs.columns[inputs.isna().any()])
inputs.Age = inputs.Age.fillna(inputs.Age.mean())
#print(inputs.head(10))

X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.3)

model = GaussianNB()
model.fit(X_train, y_train)
model.score(X_test, y_test)

print(model.predict(X_test[:10]))
print(model.predict_proba(X_test[:10]))

print(cross_val_score(GaussianNB(), X_train, y_train, cv=5))




