import pandas as pd

df = pd.read_csv("movies.csv")
#print(df.head(7))
#print(df.tail(4))
#print(df.imdb_rating.min(), df.imdb_rating.max(), df.imdb_rating.mean())
df_b = df[df.industry == "Bollywood"]
df_h = df[df.industry == "Hollywood"]

print(df_b)
print(df_h)