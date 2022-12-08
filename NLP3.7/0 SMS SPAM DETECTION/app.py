# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import r2_score

# load data
data = pd.read_csv("data/sms_spam.csv", encoding="ISO-8859-1")
data = data.loc[:, ["v2", "v1"]]

# print data
print(data.head(2))

# change column name
data = data.rename({
    "v1": "target_ctg",
    "v2": "text"
}, axis=1)

# display data
print(data.head())

# map ham : 0 & spam :1
data["target_num"] = data["target_ctg"].map({
    "ham": 0,
    "spam": 1
})

data = data.loc[:, ["text", "target_num", "target_ctg"]]
# display
print(data.head())

# converting text features into vectors using TfidfVectorizer
"""
tfid = TfidfVectorizer(decode_error="ignore")

X = tfid.fit_transform(data["data"])
print(X.shape)

y = data["target"]
print(y.head())
"""

# converting text features into vectors using CountVectorizer
count_vect = CountVectorizer(decode_error="ignore")

X = count_vect.fit_transform(data["text"])

y = data["target_num"]

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# modelling with MultinomialNB()
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# training score
print(f"Training Score : {model.score(X_train,y_train)}")
print(f"Test Score : {model.score(X_test,y_test)} ")
print(f"R2 Score : {r2_score(y_test,y_pred)} ")


# modelling with AdaBoostClassifier
model = AdaBoostClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# training score
print(f"Training Score : {model.score(X_train,y_train)}")
print(f"Test Score : {model.score(X_test,y_test)} ")
print(f"R2 Score : {r2_score(y_test,y_pred)} ")

# wordcloud visualization function


def visualize(label):
    words = ""
    for msg in data[data["target_ctg"] == label]["text"]:
        msg = msg.lower()
        words += msg + " "
    wordcloud = WordCloud(width=600, height=600).generate(words)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


# spam wordcloud visualization
visualize("spam")

# not-spam wordcloud visualization
visualize("ham")

# see what we are getting wrong
data["prediction"] = model.predict(X)

print(data["prediction"])

# things that should be spam
sneaky_spam = data[(data["prediction"] == 0) &
                   (data["target_num"] == 1)]["text"]
for msg in sneaky_spam:
    print(msg)

# things that should not be spam
not_actually_spam = data[(data["prediction"] == 1)
                         & (data["target_num"] == 0)]["text"]
for msg in not_actually_spam:
    print(msg)
