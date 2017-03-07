__future__ import division, print_function, absolute_import

import numpy as np
#import matplotlib.pyplot as plt
import tflearn
import tensorflow as tf
from random import randint
from sklearn.metrics import confusion_matrix
from tensorflow.contrib.metrics import streaming_accuracy
from tensorflow.contrib.metrics import streaming_precision
# Data loading and preprocessing
import tflearn.datasets.mnist as mnist
Images, Lables, testImages, testLables = mnist.load_data(one_hot=True)


f = randint(0,20)

x = tf.placeholder("float",[None, 784])
y = tf.placeholder("float",[None, 10])
# Building the encoder
encoder = tflearn.input_data(shape=[None, 784])
encoder = tflearn.fully_connected(encoder, 256)
encoder = tflearn.fully_connected(encoder, 64)
encoder = tflearn.fully_connected(encoder, 10, activation='softmax')

acc= tflearn.metrics.Accuracy()

# Regression, with mean square error
net = tflearn.regression(encoder, optimizer='adam', learning_rate=0.001,
                         loss='mean_square', metric=acc, shuffle_batches=True)


model = tflearn.DNN(net, tensorboard_verbose=0)

model.fit(Images, Lables, n_epoch=1, validation_set=(testImages, testLables),
          run_id="auto_encoder", batch_size=256,show_metric=True, snapshot_epoch=True)

evali= model.evaluate(testImages,testLables)
print("\n")
print("\t"+"Mean accuracy of the model is :", evali)
lables = model.predict_label(testImages)
print("\n")
print("\t"+"The predicted labels are :",lables)
prediction = model.predict(testImages)
print("\n")
print("\t"+"\t"+"\t"+"The predicted probabilities are :" )
print("\n")
print (prediction[f])
sess =tf.Session()

flattenTestLable = tf.reshape(testLables,[-1])
flattenprediction = tf.reshape(prediction,[-1])

confusionMatrix = tf.confusion_matrix(flattenTestLable, flattenprediction, num_classes =10)


with sess.as_default():
 	print("confusion matrix = ",confusionMatrix.eval())
 
recall = tf.metrics.recall(flattenprediction ,flattenTestLable, weights=None,
                     metrics_collections=None, updates_collections=None,
                     name="Recall")
precision = tf.metrics.precision(flattenTestLable,flattenprediction,weights=None, metrics_collections=None, 
								updates_collections=None, name='precision')
     print("Precision:", precision)
     print("Recall :", recall)
