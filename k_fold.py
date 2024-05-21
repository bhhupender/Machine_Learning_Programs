from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import  KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3)

# LogisticRegression
lr = LogisticRegression(solver='liblinear', multi_class='ovr')
lr.fit(X_train, y_train)
lr.score(X_test, y_test)

# SVM
svm = SVC(gamma='auto')
svm.fit(X_train, y_train)
svm.score(X_test, y_test)

# RandomForestClassifier
rf = RandomForestClassifier(n_estimators=40)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)

# KFold

kf = KFold(n_splits=3)
for train_index, test_index in kf.split([1, 2, 3, 4, 5, 6, 7, 8, 9]):
    print(train_index, test_index)


def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


folds = StratifiedKFold(n_splits=3)

scores_logistic = []
scores_svm = []
scores_rf = []

for train_index, test_index in folds.split(digits.data,digits.target):
    X_train, X_test, y_train, y_test = digits.data[train_index], digits.data[test_index], \
                                       digits.target[train_index], digits.target[test_index]
    scores_logistic.append(get_score(LogisticRegression(solver='liblinear', multi_class='ovr'), X_train, X_test, y_train, y_test))
    scores_svm.append(get_score(SVC(gamma='auto'), X_train, X_test, y_train, y_test))
    scores_rf.append(get_score(RandomForestClassifier(n_estimators=40), X_train, X_test, y_train, y_test))

print(scores_logistic)
print(scores_svm)
print(scores_rf)

# cross_val_score function
cross_val_score(LogisticRegression(solver='liblinear', multi_class='ovr'), digits.data, digits.target, cv=3)
cross_val_score(SVC(gamma='auto'), digits.data, digits.target, cv=3)
cross_val_score(RandomForestClassifier(n_estimators=40), digits.data, digits.target, cv=3)

# Parameter tuning using k fold cross validation
scores1 = cross_val_score(RandomForestClassifier(n_estimators=5), digits.data, digits.target, cv=10)
print(np.average(scores1))

scores2 = cross_val_score(RandomForestClassifier(n_estimators=20), digits.data, digits.target, cv=10)
print(np.average(scores2))

scores3 = cross_val_score(RandomForestClassifier(n_estimators=30), digits.data, digits.target, cv=10)
print(np.average(scores3))

scores4 = cross_val_score(RandomForestClassifier(n_estimators=40), digits.data, digits.target, cv=10)
print(np.average(scores4))

