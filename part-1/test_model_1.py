from pylab import * 
import sys
import caffe
import os
import cv2
import numpy 

caffe.set_device(1)
caffe.set_mode_gpu()
test_label = numpy.loadtxt('../lmdb_data/testing_label.txt')
net = caffe.Net('../caffemodel-1/lenet_test_first.prototxt','../caffemodel-1/lenet1_iter_1000.caffemodel', caffe.TEST)
#Calculating test error rate and classwise error rate
class_err = zeros(10)
class_labels = zeros(10)

#Loop checks how many samples belong to each class
for i in range(0, len(test_label)):
	class_labels[int(test_label[i])]+=1


total_err = 0
for test_itr in range(0, 10000):
        net.forward()
        if(net.blobs['score'].data.argmax(1)!=net.blobs['label'].data):
                total_err +=1
                class_err[int(net.blobs['label'].data)]+=1

print(float(total_err/100.0)), '%'
print 'Classwise error rate'
print ((class_err/class_labels)*100)
