from ReadPreprocessData import read_preprocess
from Tokenize import tokenize
from SharedFunctions import get_current_time, fmt, find_accuracy

from sklearn.preprocessing import normalize, scale
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from datetime import datetime

# reading and preprocessing data
t = get_current_time()
train_features, train_labels, test_features, test_labels = read_preprocess()
print("Time taken to Read and Preprocess Raw Data:", datetime.strptime(get_current_time(), fmt) - datetime.strptime(t, fmt))

# vectorizing data
t = get_current_time()
vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
train_features = vectorizer.fit_transform(train_features)
test_features = vectorizer.transform(test_features)
print("Time taken to Vectorize:", datetime.strptime(get_current_time(), fmt) - datetime.strptime(t, fmt))
print("Vector Length:", len(vectorizer.get_feature_names()))

# preprocessing data
t = get_current_time()
# train_features = normalize(train_features, axis=1, copy=True, return_norm=False)
# test_features = normalize(test_features, axis=1, copy=True, return_norm=False)

# train_features = scale(train_features, axis=0, with_mean=False)
# test_features = scale(test_features, axis=0, with_mean=False)
print("Time taken to Preprocess:", datetime.strptime(get_current_time(), fmt) - datetime.strptime(t, fmt))

# training
t = get_current_time()
classifier = MultinomialNB()
classifier.fit(train_features, train_labels)
print("Time taken to Fit:", datetime.strptime(get_current_time(), fmt) - datetime.strptime(t, fmt))

# fitting
t = get_current_time()
predicted_labels = classifier.predict(test_features)
print(predicted_labels)
print("Time taken to Predict:", datetime.strptime(get_current_time(), fmt) - datetime.strptime(t, fmt))

# finding Accuracy
find_accuracy(predicted_labels, test_labels)
