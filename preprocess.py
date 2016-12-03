import psana
import time, os
from pyimgalgos.MedianFilter import median_filter_ndarr
import numpy as np
import h5py

exp = 'cxic0415'
runNum = 92
inPath = '/reg/d/psdm/cxi/cxic0415/scratch/yoon82/psocake/'
outPath = '/reg/d/psdm/cxi/cxitut13/scratch/liponan/'
detname = 'DscCsPad'
truncateEvents = -1 # Set to -1 to get all images

cropT = 700;
cropB = 1050
cropL = 700;
cropR = 1050;
testSize = 5

def prepocess(eventNumber):
    evt = run.event(times[eventNumber])
    calib = det.calib(evt) * det.mask(evt, calib=True, status=True,
                                      edges=True, central=True,
                                      unbond=True, unbondnbrs=True)
    # background suppression
    medianFilterRank = 5
    calib -= median_filter_ndarr(calib, medianFilterRank)
    # crop
    img = det.image(evt, calib)[cropT:cropB, cropL:cropR]  # crop inside water ring
    return img

ds = psana.DataSource('exp='+exp+':run='+str(runNum)+':idx')
run = ds.runs().next()
times = run.times()
numEvents = len(times)
env = ds.env()
eventNumber = 615
evt = run.event(times[eventNumber])
det = psana.Detector(detname)
tic = time.time()
img = prepocess(eventNumber)
toc = time.time()
imgShape = img.shape

# Generate train/test list from quartile
fname = inPath + 'r' + str(runNum).zfill(4) + '/' + exp + '_' + str(runNum).zfill(4) + '.cxi'
f = h5py.File(fname, 'r')
numIndex = f['/entry_1/result_1/nPeaksAll'].value
if truncateEvents > -1: numIndex = numIndex[:truncateEvents]
f.close()

lowerB = np.percentile(numIndex, q=25)
higherB = np.percentile(numIndex, q=75)
print("lower,higher bounds: ", lowerB, higherB)

missInd = np.where(numIndex<=lowerB)[0]
hitInd = np.where(numIndex>=higherB)[0]
print("number of misses, hits: ", len(missInd), len(hitInd))

# Split into training / testing sets
missInd_test = missInd[-testSize:]
hitInd_test = hitInd[-testSize:]
missInd_train = missInd[:len(missInd)-testSize]
hitInd_train = hitInd[:len(hitInd)-testSize]

fname = outPath + exp + '_' + str(runNum).zfill(4) + '.h5'
if os.path.isfile(fname):
    print "Remove: ", fname
    os.remove(fname)
f = h5py.File(fname,'w')
ds_missTrain = f.create_dataset("/data/missTrain", (len(missInd_train), 1, imgShape[0], imgShape[1]), dtype='float32', chunks=(1, 1, imgShape[0], imgShape[1]))
ds_hitTrain = f.create_dataset("/data/hitTrain", (len(hitInd_train), 1, imgShape[0], imgShape[1]), dtype='float32', chunks=(1, 1, imgShape[0], imgShape[1]))
ds_missTest = f.create_dataset("/data/missTest", (len(missInd_test), 1, imgShape[0], imgShape[1]), dtype='float32', chunks=(1, 1, imgShape[0], imgShape[1]))
ds_hitTest = f.create_dataset("/data/hitTest", (len(hitInd_test), 1, imgShape[0], imgShape[1]), dtype='float32', chunks=(1, 1, imgShape[0], imgShape[1]))
ds_missTrainY = f.create_dataset("/data/missTrainY", (len(missInd_train),), dtype=int)
ds_hitTrainY = f.create_dataset("/data/hitTrainY", (len(hitInd_train),), dtype=int)
ds_missTestY = f.create_dataset("/data/missTestY", (len(missInd_test),), dtype=int)
ds_hitTestY = f.create_dataset("/data/hitTestT", (len(hitInd_test),), dtype=int)
for i, val in enumerate(missInd_train):
    print "i: ", i, val
    ds_missTrain[i, :, :, :] = prepocess(val)
    ds_missTrainY[i] = 0
    f.flush()
for i, val in enumerate(hitInd_train):
    ds_hitTrain[i, :, :, :] = prepocess(val)
    ds_hitTrainY[i] = 1
    f.flush()
for i in range(testSize):
    print "i: ", i, missInd_test[i]
    ds_missTest[i, :, :, :] = prepocess(missInd_test[i])
    ds_missTestY[i] = 0
    ds_hitTest[i, :, :, :] = prepocess(hitInd_test[i])
    ds_hitTestY[i] = 1
    f.flush()
f.close()
