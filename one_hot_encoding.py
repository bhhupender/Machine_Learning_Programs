import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


df = pd.read_csv('homepricesn.csv')
#print(df)

#use pandas to create dummy variables
dummies = pd.get_dummies(df.town)
#print(dummies)

merged = pd.concat([df, dummies], axis='columns')
#print(merged)

final = merged.drop(['town'], axis='columns')
#print(final)

#Dummy Variable Trap
#Remove one town from Final DF
final = final.drop(['west windsor'], axis='columns')
#print(final)

X = final.drop(['price'], axis='columns')
#print(X)

y = final.price

#Train the Model
model = LinearRegression()
model.fit(X, y)
#print(model.predict(X))
#print(model.predict([[3400, 0, 0]]))
#print(model.predict([[2800, 0, 1]]))

#Using sklearn OneHotEncoder
#First step is to use label encoder to convert town names into numbers

le = LabelEncoder()
dfle = df
dfle.town = le.fit_transform(dfle.town)
#print(dfle)

X = dfle[['town', 'area']].values
#print(X)

y = dfle.price.values
ct = ColumnTransformer([('town', OneHotEncoder(), [0])], remainder='passthrough')
X = ct.fit_transform(X)
#print(X)
X = X[:, 1:]
#print(X)

model.fit(X, y)
print(model.predict([[0, 1, 3400]])) #3400 sqr ft home in west windsor
print(model.predict([[1,0,2800]])) #2800 sqr ft home in robbinsville




