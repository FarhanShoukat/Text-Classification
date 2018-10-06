import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def stem(tokens):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(email):
    tokens = nltk.word_tokenize(email)
    return stem(tokens)
