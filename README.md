# ProjectTF

The more important aspect of Machine learning algorithms is the evaluation of each of them and comparing it with all the similar algorithms.

The Stacked autoencoder here will be evaluated with some scores like Accuracy, Precision, Recall, F-Score and at last create a confision matrix which will visualize all the results and confirm different score.

Here the MNIST handwritten dataset contains a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.

Four files are available on the website [http://yann.lecun.com/exdb/mnist/] : 
train-images-idx3-ubyte.gz:  training set images (9912422 bytes) 
train-labels-idx1-ubyte.gz:  training set labels (28881 bytes) 
t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes) 
t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes) 

The images are 28x28 pixels and the labels have 10 classes specifing to which class the images 

Convolutional Neural Network is also evaluated on same MNIST handwritten dataset as the autoencoder

However if some one wants to use a differet dataset, they very well can but do keep in mind that both of these examples are built using Tflearn, Tensorflow and Sklearn so they need to keep in mind the bits and peices of them.
Also while changing the data set you must keep in mind the shape and rank of the tensors. All these functions uses a specific corelation between the inputs and labels.


Other models will be updated soon.
