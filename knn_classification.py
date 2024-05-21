import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings('ignore')
iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])
df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]

X = df.drop(['target', 'flower_name'], axis='columns')
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create KNN (K Neighrest Neighbour Classifier)

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
#print(knn.score(X_test, y_test))
#print(knn.predict([[4.8, 3.0, 1.5, 0.3]]))

# Plot Confusion Matrix
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
#print(cm)

# Print classification report for precesion, recall and f1-score for each classes
print(classification_report(y_test, y_pred))

