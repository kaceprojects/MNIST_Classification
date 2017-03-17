from mnist import MNIST
import numpy
import cv2
import matplotlib
import sys
import os
import glob
import random

import caffe
from caffe.proto import caffe_pb2
import lmdb

#add gaussian noise

def gauss_noise(im, stddev):
	mean = 0
	std = stddev
	gauss = numpy.random.normal(mean, std, (row, col))
	gauss = numpy.reshape(gauss, (row, col))
	noisy_im = im+gauss
	noisy_im *= 255.0/noisy_im.max()
	noisy_im = (numpy.uint8(noisy_im))
	return noisy_im

#Package image and label into datum

def make_datum(img, label):
    #image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=1,
        width=28,
        height=28,
        label=label,
        data=numpy.rollaxis(img, 1).tobytes())

#reads user argument for standard deviation value

if(len(sys.argv)<2):
	val = 8
elif((int(sys.argv[1]) == 8) | (int(sys.argv[1])==32)| (int(sys.argv[1])==128)):
	val = int(sys.argv[1])
else:
	val = 8


train_lmdb = '../lmdb_data/train_lmdb_part2_'+ str(val)
os.system('rm -rf  ' + train_lmdb)


#Getting training images
mndata = MNIST('../training-files')
images, labels = mndata.load_training()
row = 28
col = 28

#Write to db
in_db = lmdb.open(train_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
	for i in range(len(images)):
		img = numpy.reshape(images[i], (row, col)) #Convert to numpy arrays
		im = numpy.uint8(img)
		im = gauss_noise(im, val) #add noise
		label = labels[i]
		datum = make_datum(im, label)
		in_txn.put('{:0>5d}'.format(i), datum.SerializeToString())
in_db.close()
print '\nFinished processing all images'

print "Done Adding Noise"
