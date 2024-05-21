from sklearn import svm, datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression



iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['flower'] = iris.target
df['flower'] = df['flower'].apply(lambda x: iris.target_names[x])
# print(df)
# Approach 1: Use train_test_split and manually tune parameters by trial and error
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)
model = svm.SVC(kernel='rbf', C=30, gamma='auto')
model.fit(X_train, y_train)
model.score(X_test, y_test)

# Approach 2: Use K Fold Cross validation
cross_val_score(svm.SVC(kernel='linear', C=10, gamma='auto'), iris.data, iris.target, cv=5)
cross_val_score(svm.SVC(kernel='rbf', C=10, gamma='auto'), iris.data, iris.target, cv=5)
cross_val_score(svm.SVC(kernel='rbf', C=20, gamma='auto'), iris.data, iris.target, cv=5)

# Above approach is tiresome and very manual. We can use for loop as an alternative
kernels = ['rbf', 'linear']
C = [1,10,20]
avg_scores = {}
for kval in kernels:
    for cval in C:
        cv_scores = cross_val_score(svm.SVC(kernel=kval, C=cval,gamma='auto'), iris.data, iris.target, cv=5)
        avg_scores[kval + '_' + str(cval)] = np.average(cv_scores)

# print(avg_scores)
# From above results we can say that rbf with C=1 or 10 or linear with C=1 will give best performance

# Approach 3: Use GridSearchCV
# GridSearchCV does exactly same thing as for loop above but in a single line of code

clf = GridSearchCV(
    svm.SVC(gamma='auto'), {
    'C': [1, 10, 20],
    'kernel': ['rbf', 'linear']
    },
    cv=5, return_train_score=False)
clf.fit(iris.data, iris.target)
# print(clf.cv_results_)
df = pd.DataFrame(clf.cv_results_)
#print(df[['param_C', 'param_kernel', 'mean_test_score']])
#print(clf.best_params_)
#print(clf.best_score_)

# Use RandomizedSearchCV to reduce number of iterations and with random combination of parameters.
# This is useful when you have too many parameters to try and your training time is longer.
# It helps reduce the cost of computation

rs = RandomizedSearchCV(svm.SVC(gamma='auto'), {
        'C': [1, 10, 20],
        'kernel': ['rbf', 'linear']
    },
    cv=5,
    return_train_score=False,
    n_iter=2
)
rs.fit(iris.data, iris.target)
df1 = pd.DataFrame(rs.cv_results_)[['param_C', 'param_kernel', 'mean_test_score']]
#print(df1)

# How about different models with different hyperparameters?

model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params': {
            'C': [1, 10, 20],
            'kernel': ['rbf', 'linear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [1, 5, 10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1, 5, 10]
        }
    }
}

scores = []
for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(iris.data, iris.target)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })

df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
print(df)

# Based on above, I can conclude that SVM with C=1 and kernel='rbf' is the best model for
# solving my problem of iris flower classification