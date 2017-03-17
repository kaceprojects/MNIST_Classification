from pylab import * 
import sys
import caffe
import os
import cv2
import numpy 
caffe.set_device(4) #Change according to GPU availability
caffe.set_mode_gpu()


#Reading command line argument for testing model
if(len(sys.argv)<2):
	val = 8
elif((int(sys.argv[1]) == 8) | (int(sys.argv[1])==32)| (int(sys.argv[1])==128)):
	val = int(sys.argv[1])
else:
	val = 8


test_label = numpy.loadtxt('../lmdb_data/testing_label.txt')

net = caffe.Net('../caffemodel-2/lenet_test_second.prototxt','../caffemodel-2/lenet2_'+ str(val)+'_iter_1000.caffemodel', caffe.TEST)
class_err = zeros(10)
class_labels = zeros(10)

#Loop calculates number of samples per class

for i in range(0, len(test_label)):
	class_labels[int(test_label[i])]+=1


#Calculating test error rate and classwise error rate
total_err = 0
for test_itr in range(0, 10000):
        net.forward()
        if(net.blobs['score'].data.argmax(1)!=net.blobs['label'].data):
                total_err +=1
                class_err[int(net.blobs['label'].data)]+=1

print(float(total_err/100.0)), ' %'
print 'Classwise error-rate'
print ((class_err/class_labels)*100)
