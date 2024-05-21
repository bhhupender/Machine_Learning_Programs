import pandas as pd
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA


digits = load_digits()

df = pd.DataFrame(digits.data, columns=digits.feature_names)
X = df
y = digits.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=30)

model = LogisticRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)

# Use PCA to reduce dimensions

pca = PCA(0.95)
X_pca = pca.fit_transform(X)

# print(pca.explained_variance_ratio_)
#print(pca.n_components_)

X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=30)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_pca, y_train)
print(model.score(X_test_pca, y_test))



