import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split


# load data
data = pd.read_csv("data/spambase.data")
data.head()

# split data into independent & dependent features
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# train test split manual
X_train = X.iloc[:-100]
X_test = X.iloc[-100:]
y_train = y.iloc[:-100]
y_test = y.iloc[-100:]

# train test split using sklearn
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, shuffle=True)

# MultinomialNB
# instantiate model
model = MultinomialNB()

# train model
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# model score
model_score = model.score(X_test, y_test)

print(f"Classification rate for NB : {round(model_score,2)}")


# AdaBoostClassifier

# instantiate model
model = AdaBoostClassifier()

# train model
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# model score
model_score = model.score(X_test, y_test)

print(f"Classification rate for AdaBoostClassifier : {round(model_score,2)}")
