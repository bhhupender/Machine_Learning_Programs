import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier


df = pd.read_csv('diabetes.csv')
# print(df)
# print(df.isnull().sum())
# print(df.describe())
# print(df.Outcome.value_counts())

# Train Test Split
X = df.drop("Outcome", axis='columns')
y = df.Outcome

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=10)

# Train using stand alone model
scores = cross_val_score(DecisionTreeClassifier(), X, y, cv=5)
#print(scores, scores.mean())

# Train using Bagging

bag_model = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, max_samples=0.8, oob_score=True, random_state=0)

bag_model.fit(X_train, y_train)
print(bag_model.oob_score_)
print(bag_model.score(X_test, y_test))


