import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline



df = pd.read_csv('spam.csv')
#print(df)

#print(df.groupby('Category').describe())
df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
#print(df)

X_train, X_test, y_train, y_test = train_test_split(df.Message, df.spam)
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)

model = MultinomialNB()
model.fit(X_train_count, y_train)

emails = [
    'Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]
emails_count = v.transform(emails)
#print(model.predict(emails_count))

X_test_count = v.transform(X_test)
#print(model.score(X_test_count, y_test))

# SKLearn Pipeline
clf = Pipeline(
    [
        ('vectorizer', CountVectorizer()),
        ('nb', MultinomialNB())
    ])

clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
print(clf.predict(emails))

