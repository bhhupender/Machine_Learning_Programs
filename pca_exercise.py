import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

df = pd.read_csv('heart.csv')
#print(df.head())
#print(df.describe())

# Treat Outliers

df1 = df[df.Cholesterol > (df.Cholesterol.mean()+3*df.Cholesterol.std())]
df2 = df1[df1.Oldpeak <= (df1.Oldpeak.mean()+3*df1.Oldpeak.std())]
df3 = df2[df2.RestingBP <= (df2.RestingBP.mean()+3*df2.RestingBP.std())]

df4 = df3.copy()
df4.ExerciseAngina.replace(
    {
        'N': 0,
        'Y': 1
    },
    inplace=True)

df4.ST_Slope.replace(
    {
        'Down': 1,
        'Flat': 2,
        'Up': 3
    },
    inplace=True
)

df4.RestingECG.replace(
    {
        'Normal': 1,
        'ST': 2,
        'LVH': 3
    },
    inplace=True)
#print(df4)

df5 = pd.get_dummies(df4)
#print(df5.head)

X = df5.drop("HeartDisease", axis='columns')
y = df5.HeartDisease

#print(X.head())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=30)

model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)
model_rf.score(X_test, y_test)

# Use PCA to reduce dimensions

pca = PCA(0.95)
X_pca = pca.fit_transform(X)

X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=30)
model_rfp = RandomForestClassifier()
model_rfp.fit(X_train_pca, y_train)
print(model_rfp.score(X_test_pca, y_test))

