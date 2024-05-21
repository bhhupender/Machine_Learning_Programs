import numpy as np
import pandas as pd
from statistics import mean

pd.set_option("display.precision", 2)
df = pd.read_csv("telecom_churn.csv")
df["Churn"] = df["Churn"].astype("int64")
#print(df.describe())
#print(df.shape)
#print(df.columns)
#print(df.info())
#print(df.describe(include=["object", "bool"]))
#print(df["Churn"].value_counts())
#print(df["Churn"].value_counts(normalize=True))
#print(df.sort_values(by="Total day charge", ascending=False).head())
#print(df.sort_values(by=["Churn", "Total day charge"], ascending=[True, False]).head())
#print(mean(df["Churn"]))
#print(mean(df["Churn"] == 1))
#print(df.apply(np.max))
df = df[df["State"].apply(lambda state: state[0] == "W")]
print(df.head())