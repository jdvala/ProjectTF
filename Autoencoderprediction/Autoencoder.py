""" Auto Encoder Example.
Using an Stacked auto encoder on MNIST handwritten digits, and evaluating its performance with different scores

References:
     Tflearn.org/examples
     Tensorflow.org
Links:

    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

Method and Examples Used:
[1] An simple example from Tflean, which is an higher level API for tensorflow provided with an autoencoder example which reconstructed the 
images but the motive here was to evaluate this autoencoder with different score so it could be fine tuned in future for various specific tasks.
Also for reconstructing the images this program used decoder which we don't need for our evaluation.

[2] Secondly the last layer for classification should be softmax layer and here I changed here acoordingly 

[3] I am not using Confusion matrix from tensorflow, rather I used sklearn library for that purpose.

[4] All the steps involved in this program is commented out for better understanding of this program. 

By: Jay Vala with help from Navin F Henry 
"""
from __future__ import division, print_function, absolute_import

import numpy as np
import tflearn
import tensorflow as tf
from random import randint
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Data loading and preprocessing
import tflearn.datasets.mnist as mnist
Images, Lables, testImages, testLables = mnist.load_data(one_hot=True)

# Random integer to restrict the results 
f = randint(0,20)

# Placeholders to hold data before feeding it into network
x = tf.placeholder("float",[None, 784]) 	#for images with shape of None,784
y = tf.placeholder("float",[None, 10])		#for lables with shape of None,10

# Building the encoder
encoder = tflearn.input_data(shape=[None, 784])
encoder = tflearn.fully_connected(encoder, 256)
encoder = tflearn.fully_connected(encoder, 64)
encoder = tflearn.fully_connected(encoder, 10, activation='softmax')

#For calculating Accuracy at every step of model training 
acc= tflearn.metrics.Accuracy()

# Regression, with mean square error (learn about it more here http://tflearn.org/layers/estimator/)
net = tflearn.regression(encoder, optimizer='adam', learning_rate=0.001,
                         loss='mean_square', metric=acc, shuffle_batches=True)


# Mpdeling the Neural Network (for details http://tflearn.org/models/dnn/)
model = tflearn.DNN(net, tensorboard_verbose=0)

# Training the Neural Network (for details http://tflearn.org/models/dnn/)
model.fit(Images, Lables, n_epoch=1, validation_set=(testImages, testLables),
          run_id="auto_encoder", batch_size=256,show_metric=True, snapshot_epoch=True)

# Here I evaluate the model with Test Images and Test Lables, calculating the Mean Accuracy of the model.
evaluation= model.evaluate(testImages,testLables)
print("\n")
print("\t"+"Mean accuracy of the model is :", evaluation)

# Prediction the Lables of the Images that we give to the model just to have a clear picture of Neural Netwok
lables = model.predict_label(testImages)
print("\n")
print("\t"+"The predicted labels are :",lables)

# Predicted probailites 
y = model.predict(testImages)
print("\n")
print("\t"+"\t"+"\t"+"The predicted probabilities are :" )
print("\n")
print (y[f])

# Running a session to feed calculate the confusion matrix
sess = tf.Session()
# taking the argumented maximum of the predicted probabilities for generating confusion matrix 
prediction = tf.argmax(y,1)
# displaying length of predictions and evaluating them in a session 
with sess.as_default():
	print (len(prediction.eval()))
	predicted_labels = prediction.eval()
# Again importing the mnist data with one hot as false because we need to know the truepositive and other values for evaluation
Images, Lables, testImages, targetLables = mnist.load_data(one_hot=False)

# Used Sklearn library for evaluation as tensorflows library was not documented properly 
# Generated the Confusion Matrix 
confusionMatrix = confusion_matrix(targetLables, predicted_labels)
print (confusionMatrix)

# Classification_report in Sklearn provide all the necessary scores needed to succesfully evaluate the model. 
classification = classification_report(targetLables,predicted_labels, digits=4, 
				target_names =['class 0','class 1','class 2','class 3','class 4','class 5','class 6','class 7','class 8','class 9'])
print (classification)





