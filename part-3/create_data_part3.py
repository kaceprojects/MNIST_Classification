import matplotlib
matplotlib.use('Agg')
from mnist import MNIST
import numpy
import sys
import os
import random
import cv2
import caffe
from caffe.proto import caffe_pb2
import lmdb

# Package data and label into a datum
def make_datum(img, label):
    return caffe_pb2.Datum(
        channels=1,
        width=28,
        height=28,
        label=label,
        data=numpy.rollaxis(img, 1).tobytes())


# Read command line arguments
if(len(sys.argv)<2):
	val = 5	
elif (int(sys.argv[1])>100):
	val = 5
else:	
	val = int(sys.argv[1])

# Loading mnist data

train_lmdb = '../lmdb_data/train_lmdb_part3_' + str(val)
mndata = MNIST('../training-files')
images, labels = mndata.load_training()
row = 28
col = 28

#randomize labels
print 'Percentage of randomization', val
num = (val* len(labels))//100
idx = random.sample(range(0, len(labels)), num)
for i in idx:
	labels[i] = random.randint(0, 9)

# Storing new labels of train data

filename = '../lmdb_data/training_label_random_' + str(val) + '.txt'
numpy.savetxt(filename, labels, delimiter = ',')

print "Creating lmdb"

# store in lmdb database
in_db = lmdb.open(train_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
	for i in range(len(images)):
		img = numpy.reshape(images[i], (row, col)) #Create numpy arrays to represent image
		im = numpy.uint8(img)
		label = labels[i]
		datum = make_datum(im, label)
		in_txn.put('{:0>5d}'.format(i), datum.SerializeToString())
in_db.close()

print '\nFinished processing all images'












