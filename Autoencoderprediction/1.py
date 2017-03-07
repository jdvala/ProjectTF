from __future__ import division, print_function, absolute_import

import numpy as np
#import matplotlib.pyplot as plt
import tflearn
import tensorflow as tf
from random import randint
from sklearn.metrics import confusion_matrix

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

model.fit(Images, Lables, n_epoch=1, validation_set=(testImages, testLables),
          run_id="auto_encoder", batch_size=256,show_metric=True, snapshot_epoch=True)

#Applying the ablove model on test Images and evaluating as well as prediction the lables
evali= model.evaluate(testImages,testLables)
print("\n")
print("\t"+"Mean accuracy of the model is :", evali)
lables = model.predict_label(testImages)
print("\n")
print("\t"+"The predicted labels are :",lables[f])
prediction = model.predict(testImages)
print("\n")
print("\t"+"\t"+"\t"+"The predicted probabilities are :" )
print("\n")
print (prediction[f])

softmaz = tf.nn.softmax (prediction, name='softmax')

sess =tf.Session()
with sess.as_default():
	evaluate1 = softmaz.eval()
	print("\t"+"softmax logits")
	print("\n")
	print(evaluate1)

rankOfSoftmax = tf.rank(evaluate1)	
rankOfTestlables =tf.rank(testLables)
ShapeOfSoftmax = tf.shape(evaluate1)
ShapeOfTestLables = tf.shape(testLables)
# we have real lables as testLables, we have logits as predictions we can create confusion matrix
print(testLables)
print("\n")
with sess.as_default():
 print("\t"+"Rank of Softmax layer's output is :",rankOfSoftmax.eval())
 print("\n")
 print("\t"+"Rank of Testlables is:",rankOfTestlables.eval())
 print("\n")
 print("\t"+"Shape of softmax layer is :",ShapeOfSoftmax.eval())
 print("\n")
 print("\t"+"Shape of TestLables is:", ShapeOfTestLables.eval())
 print("\n")
j = type(evaluate1)
print ("\t"+"type of softmax",j)
print("\n")
k = type(testLables)
print("\t"+"type of testLables", k)
print("\n")

flattenTestLable = tf.reshape(testLables,[-1])
flattenSoftmax =tf.reshape(softmaz,[-1])
shapetest = tf.TensorShape(flattenTestLable)
shapeSoftmax = tf.TensorShape(flattenSoftmax)
print ("\t"+"The lenght of flatten Test Lables is",shapetest)
print ("\t"+"The lenght of flatten softmax layer is ", shapeSoftmax)

confusionMatrix = tf.confusion_matrix(flattenTestLable, flattenSoftmax)
with sess.as_default():
	print("\t"+"confusion Matrix is :",confusionMatrix.eval()) 





#casted = tf.cast(testLables, tf.float32)	


#with sess.as_default():
 #print("Casted testLables:", casted.eval())

#confusionMatrix = confusion_matrix(testLables, softmaz)
#print(confusionMatrix)



    # Step 1:
    # Let's create 2 vectors that will contain boolean values, and will describe our labels

#is_label_one = tf.cast(testLables, dtype=tf.bool)
#is_label_zero = tf.logical_not(is_label_one)
    # Imagine that labels = [0,1]
    # Then
    # is_label_one = [False,True]
    # is_label_zero = [True,False]

    # Step 2:
    # get the prediction and false prediction vectors. correct_prediction is something that you choose within your model.
#correct_prediction = tf.nn.in_top_k(softmaz, testLables, 1, name="correct_answers")
#print (correct_prediction)
#false_prediction = tf.logical_not(correct_prediction)

    # Step 3:
    # get the 4 metrics by comparing boolean vectors
    # TRUE POSITIVES
#true_positives = tf.reduce_sum(tf.to_int32(tf.logical_and(correct_prediction,is_label_one)))

    # FALSE POSITIVES
#false_positives = tf.reduce_sum(tf.to_int32(tf.logical_and(false_prediction, is_label_zero)))

    # TRUE NEGATIVES
#true_negatives = tf.reduce_sum(tf.to_int32(tf.logical_and(correct_prediction, is_label_zero)))

    # FALSE NEGATIVES
#false_negatives = tf.reduce_sum(tf.to_int32(tf.logical_and(false_prediction, is_label_one)))
	
# Now you can do something like this in your session:

#true_positives, \
#false_positives, \
#true_negatives, \
#false_negatives = sess.run(evaluation(softmaz,testLabels))

# you can print the confusion matrix using the 4 values from above, or get precision and recall:
#precision = float(true_positives) / float(true_positives+false_positives)
#recall = float(true_positives) / float(true_positives+false_negatives)
#print ("Precision", precision)
#print("recall", recall)
# or F1 score:
#F1_score = 2 * ( precision * recall ) / ( precision+recall )
#print("F1 socre", F1_score)

























































#precision = tf.metrics.precision(testLables, evaluate1)
#conf = confusion_matrix(testLables, evaluate1)
#print ("COnfusion_matrix:::::",conf)

#init = tf.global_variables_initializer()
#with tf.Session() as sess:
 #   sess.run(init)
  #  for x in range (0,55000):
   #     recall1 = tf.streaming_recall(prediction,lables, weights=None,
    #                 metrics_collections=None, updates_collections=None,
     #                name="Recall")
      #  print("recall =",recall)

