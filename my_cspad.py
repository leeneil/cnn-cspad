from __future__ import absolute_import
from __future__ import print_function

import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import matplotlib.pyplot as plt
import pylab as pl
import matplotlib.cm as cm
import numpy as np
np.random.seed(1337) # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

import theano
print(theano.config.device)

import psana
import h5py


f = h5py.File('/reg/d/psdm/cxi/cxitut13/res/yoon82/cxis0813/cxis0813_0032.cxi','r')
imgs = f['/entry_1/data_1/data']
hits = f['/entry_1/result_1/nPeaks'].value
# f.close()
#print(run)

# input image dimensions
# img.shape
#plt.imshow(imgs[1500,:,:],interpolation='none',vmax=50,vmin=0)
#plt.show()


# override the real # of events
numEvents = imgs.shape[0]

numTrain = 5
numTest = 100

numIters = 10
nb_classes = 2
batch_size = 10
nb_epoch = 2

print(imgs.shape)
print(imgs.dtype)
print(type(imgs))

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(1480,1552,1)))
# The Dropout is not in the original keras example, it's just here to demonstrate how to
# correctly handle train/predict phase difference when visualizing convolutions below
model.add(Dropout(0.1))
convout1 = Activation('relu')
model.add(convout1)
model.add(Convolution2D(32, 3, 3))

convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])



X_train = np.empty([numTrain, imgs.shape[1], imgs.shape[2], 1])
X_test = np.empty([numTest, imgs.shape[1], imgs.shape[2], 1])


for t in range(0,numIters):

	randList = np.random.permutation(numEvents)
	trainList = randList[:numTrain]
	testList = randList[numTrain:(numTrain+numTest)]

	print(randList)
	print(trainList)
	print(testList)





	for u in range(0, numTrain):
	    n = trainList[u]
	    img = imgs[n,:,:]
	    img_reshape = np.reshape(img, [imgs.shape[1], imgs.shape[2], 1])
	    X_train[u,:,:,:] = img_reshape

	#print(X_train.shape)    
	    
	for u in range(0, numTest):
	    n = testList[u]
	    img = imgs[n,:,:]
	    img_reshape = np.reshape(img, [imgs.shape[1], imgs.shape[2], 1])
	    X_test[u,:,:,:] = img_reshape  

	# print(X_test.shape)    

	y_train = hits[trainList]
	# print(y_train)
	y_train = y_train > 50
	y_train = 1 * y_train 
	# print(y_train)


	y_test = hits[testList]
	# print(y_test)
	y_test = y_test > 50
	y_test = 1 * y_test 
	# print(y_test)

	# plt.plot(y_train,'^')
	# plt.show()

	# plt.plot(y_test,'v')
	# plt.show()

	# plt.plot(indexed,'x')
	# plt.show()




	Y_train = np_utils.to_categorical(y_train, nb_classes)

	Y_test = np_utils.to_categorical(y_test, nb_classes)


	# print(X_train.shape[1:])




	model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
	    verbose=1, validation_data=(X_test, Y_test))

	WEIGHTS_FNAME = '/reg/d/psdm/cxi/cxitut13/scratch/liponan/ml/cspad_cnn_weights_v1.hdf'
	model.save_weights(WEIGHTS_FNAME, overwrite=True)
	score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)

	print(score)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])


f.close()

