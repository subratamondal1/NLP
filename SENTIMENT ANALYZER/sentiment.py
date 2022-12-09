### import libraries ###
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
nltk.download()

# ml model
from sklearn.linear_model import LogisticRegression

# it turns word into their base form like dogs to dog, cats to cat, jumping to jump, etc
# it's purpose is to redue the vocabulary size
wordnet_lemmatizer = WordNetLemmatizer()

# stopwords such as the, a, an, etc
stopwords = set(w.rstrip() for w in open("data/stopwords.txt"))

# extracting positive reviews
positive_reviews = BeautifulSoup(
    open("data/product/electronics/positive.review").read(), features="html.parser")

positive_reviews = positive_reviews.find_all("review_text")

# no. of positive_reviews
print(len(positive_reviews))

# extracting negative reviews
negative_reviews = BeautifulSoup(
    open("data/product/electronics/negative.review").read(), features="html.parser")

negative_reviews = negative_reviews.find_all("review_text")

# no. of negative_reviews
print(len(negative_reviews))

# making length of positive reviews == length of negative reviews using shuffling
np.random.shuffle(positive_reviews)
positive_reviews = positive_reviews[:len(negative_reviews)]

# my tokenizer


def my_tokenizer(input_string):
    input_string = input_string.lower()
    tokens = nltk.tokenize.word_tokenize(input_string)
    # keep any token whose length > 2
    tokens = [t for t in tokens if len(t) > 2]

    # convert words to their base form
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]

    # remove stopwords
    tokens = [t for t in tokens if t not in stopwords]


# word to index mapper
word_to_index_map = dict()
current_index = 0

for review in positive_reviews:
    tokens = my_tokenizer(review.text)

    for token in tokens:
        if token not in word_to_index_map:
            word_to_index_map[token] = current_index
            current_index += 1
            
