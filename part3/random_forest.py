from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

## Modified from tensorflow forest contrib

## Random Forest Classifier

hparams = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
        num_trees=3, max_nodes=1000, num_classes=3, num_features=4)
classifier = tf.contrib.learn.TensorForestEstimator(hparams)

iris = tf.contrib.learn.datasets.load_iris()
data = iris.data.astype(np.float32)
target = iris.target.astype(np.float32)

monitors = [tf.contrib.learn.TensorForestLossMonitor(10, 10)]
classifier.fit(x=data, y=target, steps=100, monitors=monitors)
classifier.evaluate(x=data, y=target, steps=10)

## Random Forest Regressor

hparams = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
        num_trees=3, max_nodes=1000, num_classes=1, num_features=13,
        regression=True)

regressor = tf.contrib.learn.TensorForestEstimator(hparams)

boston = tf.contrib.learn.datasets.load_boston()
data = boston.data.astype(np.float32)
target = boston.target.astype(np.float32)

monitors = [tf.contrib.learn.TensorForestLossMonitor(10, 10)]
regressor.fit(x=data, y=target, steps=100, monitors=monitors)
regressor.evaluate(x=data, y=target, steps=10)
