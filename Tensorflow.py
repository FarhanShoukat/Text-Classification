from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.preprocessing import Normalizer
from SharedFunctions import get_current_time, fmt, find_accuracy
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from ReadPreprocessData import read_preprocess
from Tokenize import tokenize
import numpy as np
import tensorflow as tf
import scipy.sparse as ss

tf.logging.set_verbosity(tf.logging.INFO)

vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english')

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


def cnn_model_fn(features, labels, mode):



  # Dense Layer

  dense = tf.layers.dense(inputs=features, units=0, activation=tf.nn.relu)

  dense1 = tf.layers.dense(inputs=dense, units=1000, activation=tf.nn.relu)

  dense2 = tf.layers.dense(inputs=dense1, units=200, activation=tf.nn.relu)



  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dense2, units=2)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):




  # Load training and eval data

  # reading and preprocessing data
  t = get_current_time()
  train_emails, train_labels, test_emails, test_labels = read_preprocess()
  print("Time taken to Read and Preprocess Data:",
        datetime.strptime(get_current_time(), fmt) - datetime.strptime(t, fmt))

  # vectorizing data
  t = get_current_time()
  train_features = vectorizer.fit_transform(train_emails)
  test_features = vectorizer.transform(test_emails)
  print("Time taken to Vectorize:", datetime.strptime(get_current_time(), fmt) - datetime.strptime(t, fmt))
  print(len(vectorizer.get_feature_names()))


  train_data = train_features.toarray()
  eval_data = test_features.toarray()







  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")



  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x=train_data[0:23687],
      y=np.array(train_labels),
      batch_size=1,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=500)

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x=eval_data[0:23687],
      y=np.array(test_labels),
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


  results=mnist_classifier.predict(eval_input_fn)


  predictions = list(p["classes"] for p in results)

  a=np.asarray(predictions)

  np.savetxt("result.csv", np.dstack((np.arange(1, a.size + 1), a))[0], "%d,%d", header="ID,Label",comments="")


if __name__ == "__main__":
  tf.app.run()