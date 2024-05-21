import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.linear_model import  Ridge
import warnings


warnings.filterwarnings('ignore')
df = pd.read_csv('Melbourne_housing_FULL.csv')
# print(df.head())

# print(df.nunique())
# let's use limited columns which makes more sense for serving our purpose

cols_to_use = ['Suburb', 'Rooms', 'Type', 'Method', 'SellerG', 'Regionname', 'Propertycount',
               'Distance', 'CouncilArea', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Price']

df = df[cols_to_use]
# print(df.head())
# Checking for Nan values
# print(df.isna().sum())

# Handling Missing values
# Some feature's missing values can be treated as zero (another class for NA values or absence of that feature)
# like 0 for Propertycount, Bedroom2 will refer to other class of NA values
# like 0 for Car feature will mean that there's no car parking feature with house
cols_to_fill_zero = ['Propertycount', 'Distance', 'Bedroom2', 'Bathroom', 'Car']
df[cols_to_fill_zero] = df[cols_to_fill_zero].fillna(0)

# other continuous features can be imputed with mean for faster results since our focus is on Reducing overfitting
# using Lasso and Ridge Regression

df['Landsize'] = df['Landsize'].fillna(df.Landsize.mean())
df['BuildingArea'] = df['BuildingArea'].fillna((df.BuildingArea.mean()))

# Drop NA values of Price, since it's our predictive variable we won't impute it

df.dropna(inplace=True)

# Let's one hot encode the categorical features

df = pd.get_dummies(df, drop_first=True)
#print(df.head())

# Let's bifurcate our dataset into train and test dataset

X = df.drop('Price', axis='columns')
y = df['Price']

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=2)

# Let's train our Linear Regression Model on training dataset and check the accuracy on test set

#reg = LinearRegression().fit(train_X, train_y)

# print(reg.score(test_X, test_y))
# print(reg.score(train_X, train_y))

# Here training score is 68% but test score is 13.85% which is very low
# Normal Regression is clearly overfitting the data, let's try other models

# Using Lasso (L1 Regularized) Regression Model
lasso_reg = linear_model.Lasso(alpha=50, max_iter=100, tol=0.1)
lasso_reg.fit(train_X, train_y)

print(lasso_reg.score(test_X, test_y))
print(lasso_reg.score(train_X, train_y))

# Using Ridge (L2 Regularized) Regression Model

ridge_reg = Ridge(alpha=50, max_iter=100, tol=0.1)
ridge_reg.fit(train_X, train_y)

print("L2 Regularized")
print(ridge_reg.score(test_X, test_y))
print(ridge_reg.score(train_X, train_y))

# We see that Lasso and Ridge Regularizations prove to be beneficial when our Simple Linear
# Regression Model overfits. These results may not be that contrast but significant in most
# cases.Also that L1 & L2 Regularizations are used in Neural Networks too





