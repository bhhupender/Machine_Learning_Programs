import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer



df = pd.read_csv('carprices.csv')
#print(df)

dummies = pd.get_dummies(df['Car Model'])
#print(dummies)

merged = pd.concat([df, dummies], axis='columns')
#print(merged)

final = merged.drop(['Car Model', 'Mercedez Benz C class'], axis='columns')
#print(final)

X = final.drop(['Sell Price($)'], axis='columns')
#print(X)

y = final['Sell Price($)']

model = LinearRegression()
model.fit(X, y)
#print(model.score(X, y))

#Price of mercedez benz that is 4 yr old with mileage 45000
print(model.predict([[45000, 4, 0, 0]]))

#Price of BMW X5 that is 7 yr old with mileage 86000
print(model.predict([[86000, 7, 0, 1]]))




