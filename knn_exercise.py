from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd

digits = load_digits()

df = pd.DataFrame(digits.data, digits.target)
#print(df.head())
df['target'] = digits.target

X_train, X_test, y_train, y_test = train_test_split(df.drop('target',axis='columns'), df.target, test_size=0.3, random_state=10)

# Create KNN (K Neighrest Neighbour Classifier)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print(knn.score(X_test, y_test))

# Plot Confusion Matrix
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Print classification report for precesion, recall and f1-score for each classes
print(classification_report(y_test, y_pred))