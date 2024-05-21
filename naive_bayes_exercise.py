import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import  train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB

wine = load_wine()

df = pd.DataFrame(wine.data, columns=wine.feature_names)
#print(df)
#print(data.target[[10, 80, 140]])

df['target'] = wine.target

X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state=100)

model = GaussianNB()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

mn = MultinomialNB()
mn.fit(X_train, y_train)
print(mn.score(X_test, y_test))
