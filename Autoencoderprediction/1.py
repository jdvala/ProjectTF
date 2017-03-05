From __future__ import division, print_function, absolute_import

import numpy as np
#import matplotlib.pyplot as plt
import tflearn
import tensorflow as tf
from random import randint
from tensorflow.contrib import metrics as ms 
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
encoder = tflearn.fully_connected(encoder, 10)

acc= tflearn.metrics.Accuracy()

# Regression, with mean square error
net = tflearn.regression(encoder, optimizer='adam', learning_rate=0.001,
                         loss='mean_square', metric=acc, shuffle_batches=True)


model = tflearn.DNN(net, tensorboard_verbose=0)

model.fit(Images, Lables, n_epoch=20, validation_set=(testImages, testLables),
          run_id="auto_encoder", batch_size=256,show_metric=True)

#Applying the ablove model on test Images and evaluating as well as prediction the lables

evali= model.evaluate(testImages,testLables)
print("Accuracy of the model is :", evali)
lables = model.predict_label(testImages)
print("The predicted labels are :",lables[f])
prediction = model.predict(testImages)
print("The predicted probabilities are :", prediction[f])
