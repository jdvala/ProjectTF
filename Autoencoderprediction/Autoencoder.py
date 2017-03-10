from __future__ import division, print_function, absolute_import

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

y = model.predict(testImages)
print("\n")
print("\t"+"\t"+"\t"+"The predicted probabilities are :" )
print("\n")
#print (prediction[10])
sess = tf.Session()
prediction = tf.argmax(y,1)
#classification = sess.run(tf.argmax(prediction), feed_dict={x: [testImages]})
with sess.as_default():
	print (len(prediction.eval()))
	predicted_labels = prediction.eval()


Images, Lables, testImages, targetLables = mnist.load_data(one_hot=False)
confusionMatrix = confusion_matrix(targetLables, predicted_labels)
print (confusionMatrix)





#flattenTestLable = tf.reshape(testLables,[-1])
#flattenprediction = tf.reshape(prediction,[-1])


 
#                     metrics_collections=None, updates_collections=None,
                     #name="Recall")
#precision = tf.metrics.precision(flattenTestLable,flattenprediction,weights=None, metrics_collections=None, 
								#updates_collections=None, name='precision')
#print("Precision:", precision)
#print("Recall :", recall)

#import tflearn.datasets.mnist as mnist
#Images, Lables, testImages, targetLables = mnist.load_data(one_hot=False)
#with sess.as_default():

#	print ("shape of test labels ",tf.shape(testLables).eval())
#	print("shape of predicted labels",tf.shape(lables).eval())
#confusionMatrix = confusion_matrix(targetLables, lables )
#with sess.as_default():
