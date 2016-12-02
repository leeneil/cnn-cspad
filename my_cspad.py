from __future__ import absolute_import
from __future__ import print_function

from IPython import embed
import psana
import h5py
import sys, os
import time
#os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu,floatX=float32'
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import matplotlib.pyplot as plt
import pylab as pl
import matplotlib.cm as cm
import numpy as np
np.random.seed(1337) # for reproducibility


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
import theano
print(theano.config.device)

# build KERAS model
print("building CNN model")

model = Sequential()

model.add(Convolution2D(4, 5, 5, border_mode='valid', input_shape=(368,368,1)))
# The Dropout is not in the original keras example, it's just here to demonstrate how to
# correctly handle train/predict phase difference when visualizing convolutions below
# model.add(Dropout(0.1))
# model.add()
model.add(BatchNormalization(mode=0))

convout1 = Activation('relu')
model.add(convout1)

model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Convolution2D(4, 5, 5))

# TODO: Visualize the convolution layer weights

model.add(BatchNormalization(mode=0))

convout2 = Activation('relu')
model.add(convout2)

model.add(MaxPooling2D(pool_size=(5, 5)))
# model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(16))

model.add(BatchNormalization(mode=1))

model.add(Activation('relu'))
# model.add(Dropout(0.5))

model.add(Dense(nb_classes))

model.add(BatchNormalization(mode=1))

model.add(Activation('relu'))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.summary()

# Load H5 file
print("loading H5 file")

tic = time.time()
# f = h5py.File('/reg/d/psdm/cxi/cxitut13/res/yoon82/r0010/cxitut13_0010.cxi','r')
# f = h5py.File('/reg/d/psdm/cxi/cxitut13/res/yoon82/cxis0813/cxis0813_0032.cxi','r')
f = h5py.File('/reg/d/psdm/cxi/cxic0415/scratch/yoon82/psocake/r0099/cxic0415_0099.cxi', 'r')
# f = h5py.File('/dev/shm/lipon/cxis0813_0032.cxi','r')

#imgs = f['/entry_1/data_1/data']
numIndex = f['/entry_1/result_1/nPeaksAll'].value
#numIndex = numIndex[ numIndex!=-2 ]
f.close()
toc = time.time()
print("Time to read in hdf5: ",toc-tic)

print(numIndex)

ds = psana.DataSource('exp=cxic0415:run=99:idx')
run = ds.runs().next()
times = run.times()
numEvents = len(times)
env = ds.env()
eventNumber = 615
evt = run.event(times[eventNumber])
det = psana.Detector('DscCsPad')
tic = time.time()
calib = det.calib(evt) * det.mask(evt, calib=True, status=True,
                                           edges=True, central=True,
                                           unbond=True, unbondnbrs=True)
# background suppression
from pyimgalgos.MedianFilter import median_filter_ndarr
calib = det.calib(evt)
medianFilterRank = 5
calib -= median_filter_ndarr(calib, medianFilterRank)

# TODO: Perhaps try binarization

cropT = 700;
cropB = 1050
cropL = 700;
cropR = 1050;

img = det.image(evt,calib)[cropT:cropB, cropL:cropR] # crop inside water ring
imgShape = img.shape
toc = time.time()
print("Time for preprocessing per image: ",toc-tic)
plt.imshow(img)
plt.show()

# Generate train/test list from quartile
lowerB = np.percentile(numIndex,q=25)
higherB = np.percentile(numIndex,q=75)
print("lower,higher bounds: ", lowerB, higherB)

missInd = np.where(numIndex<=lowerB)[0]
hitInd = np.where(numIndex>=higherB)[0]
print("number of misses, hits: ", len(missInd), len(hitInd))

# Split into training / testing sets
testSize = 50
# TODO: add assert 50
missInd_test = missInd[-testSize:]
hitInd_test = hitInd[-testSize:]
missInd_train = missInd[:len(missInd)-testSize]
hitInd_train = hitInd[:len(hitInd)-testSize]

trainSize = len(missInd_train) + len(hitInd_train)
testSize = 2 * testSize

trainStack = np.zeros((trainSize,imgShape[0],imgShape[1]))
testStack = np.zeros((testSize,imgShape[0],imgShape[1]))
# interleave misses and hits

# Generate trainLabel, testLabel

# Visually check labels are consistent with the images

embed()

sys.exit(1)

# f.close()
#print(run)

# input image dimensions
# img.shape
# plt.imshow(imgs[1500,:,:],interpolation='none',vmax=100,vmin=0)
# plt.show()


numEvents = imgs.shape[0]

thr = 0

numTrain = 100 # TODO: Set to trainSize
numTest = 100 # TODO: Set to testSize

numIters = 1
nb_classes = 2
batch_size = 1
nb_epoch = 1

print(imgs.shape)
print(imgs.dtype)
print(type(imgs))



X_train = np.empty([numTrain, imgs.shape[1], imgs.shape[2], 1])
X_test = np.empty([numTest, imgs.shape[1], imgs.shape[2], 1])


y_train = np.empty([numTrain, 1])
y_test = np.empty([numTest, 1])

for t in range(0,numIters):
	# prepare training set 
	for u in range(0, numTrain):
	    if u % 2 == 0: 
	    	evt = run.event(times[ missInd_train[(u+0)/2] ])
	    	img = det.image(evt,calib)[cropT:cropB, cropL:cropR]
	   		y_train[u] = 0
	    else:
	    	evt = run.event(times[ hitInd_train[(u+1)/2] ])
	    	img = det.image(evt,calib)[cropT:cropB, cropL:cropR]
	    	y_train[u] = 1
	    ### sample-wise normalization ###
	    # std = np.std(img)
	    # mu = np.mean(img)
	    # img = (img-mu) / std
	    #################################
	    img_reshape = np.reshape(img, [img.shape[0], img.shape[1], 1])
	    X_train[u,:,:,:] = img_reshape
 
	# prepare testing set	    
	for u in range(0, numTest):
	    if u % 2 == 0: 
	    	evt = run.event(times[ missInd_test[(u+0)/2] ])
	    	img = det.image(evt,calib)[cropT:cropB, cropL:cropR]
	    	y_test[u] = 0
	    else:
	    	evt = run.event(times[ hitInd_test[(u+1)/2] ])
	    	img = det.image(evt,calib)[cropT:cropB, cropL:cropR]
	    	y_test[u] = 1
	    ### sample-wise normalization ###
	    # std = np.std(img)
	    # mu = np.mean(img)
	    # img = (img-mu) / std
	    #################################
	    img_reshape = np.reshape(img, [img.shape[0], img.shape[1], 1])
	    X_test[u,:,:,:] = img_reshape


	print(y_train)
	print("% of hit in training set: ", np.sum(y_train)/len(y_train))

	print(y_test)
	print(np.sum(y_test))
	print("% of hit in testing set: ", np.sum(y_test)/len(y_test))


	# plt.plot(y_train,'^')
	# plt.show()

	# plt.plot(y_test,'v')
	# plt.show()

	# plt.plot(indexed,'x')
	# plt.show()

	Y_train = np_utils.to_categorical(y_train, nb_classes)

	Y_test = np_utils.to_categorical(y_test, nb_classes)


	# print(X_train.shape[1:])




	output = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
	    verbose=1, validation_data=(X_test, Y_test))

	print(output)

	# WEIGHTS_FNAME = '/reg/d/psdm/cxi/cxitut13/scratch/liponan/ml/cspad_cnn_weights_v1.hdf'
	# model.save_weights(WEIGHTS_FNAME, overwrite=True)
	# score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)

	# print(score)
	# print('Test score:', score[0])
	# print('Test accuracy:', score[1])


f.close()

