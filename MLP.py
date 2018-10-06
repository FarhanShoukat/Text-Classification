from sklearn.neural_network import MLPClassifier

from ReadPreprocessData import read_preprocess
from Tokenize import tokenize
from SharedFunctions import get_current_time, fmt, find_accuracy

from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from datetime import datetime

# reading and preprocessing data
t = get_current_time()
train_emails, train_labels, test_emails, test_labels = read_preprocess()
print("Time taken to Read and Preprocess Data:", datetime.strptime(get_current_time(), fmt) - datetime.strptime(t, fmt))

# vectorizing data
t = get_current_time()
vectorizer = CountVectorizer(tokenizer=tokenize, stop_words='english')
train_features = vectorizer.fit_transform(train_emails)
test_features = vectorizer.transform(test_emails)
print("Time taken to Vectorize:", datetime.strptime(get_current_time(), fmt) - datetime.strptime(t, fmt))

# training
t = get_current_time()
classifier = MLPClassifier(solver='lbfgs', alpha=0.001,hidden_layer_sizes=(55,10,5,))

classifier.fit(train_features, train_labels)
print("Time taken to Fit:", datetime.strptime(get_current_time(), fmt) - datetime.strptime(t, fmt))

# fitting
t = get_current_time()
predicted_labels = classifier.predict(test_features)
print("Time taken to Predict:", datetime.strptime(get_current_time(), fmt) - datetime.strptime(t, fmt))

# finding Accuracy
find_accuracy(predicted_labels, test_labels)
