import pandas as pd
import numpy as np
from sklearn import linear_model
import pickle
import joblib




df = pd.read_csv('homeprices.csv')
model = linear_model.LinearRegression()
model.fit(df[['area']], df.price)
#print(model.coef_)
#print(model.intercept_)
#print(model.predict([[5000]]))
#savetrainedmodel
with open('model_pickle', 'wb') as file:
    pickle.dump(model, file)

#LoadSavedModel
with open('model_pickle', 'rb') as file:
    mp = pickle.load(file)
    #print(mp.predict([[5000]]))

#Use JOBLIB to save trained model
joblib.dump(model, 'model_joblib')
mj = joblib.load('model_joblib')
print(mp.predict([[5000]]))

