import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import * 
import numpy
import sys
import caffe
import os
import cv2

caffe.set_device(1)
caffe.set_mode_gpu()
class_labels = zeros(10)

test_label = numpy.loadtxt('../lmdb_data/testing_label.txt')
for i in range(0, len(test_label)):
	class_labels[int(test_label[i])]+=1

values = [5, 15, 50] # edit to run any or all models
error_rate = []
print len(values), values[0], values[1], values[2]

for i in range(0, len(values)):
	net = caffe.Net('../caffemodel-3/lenet_test_third.prototxt','../caffemodel-3/lenet3_'+ str(values[i])+'_iter_1000.caffemodel', caffe.TEST)
#Calculating test error rate and classwise error rate
	class_err = zeros(10)

	total_err = 0
	for test_itr in range(0, 10000):
	        net.forward()
	        if(net.blobs['score'].data.argmax(1)!=net.blobs['label'].data):
	                total_err +=1
	                #print(int(solver.test_nets[0].blobs['label'].data))
	                class_err[int(net.blobs['label'].data)]+=1
	#               print("Error", test_itr)
	#       print(test_itr)

	print(float(total_err/100.0))
	error_rate.append(float(total_err/100.0))
	print (class_err/class_labels)
	print i
print "Plotting"
#plotting values
plt.scatter(values, error_rate)
plt.xlabel('Percentage of Randomization')
plt.ylabel('Error-Rate')
plt.title('Error-Rate as a function of Randomized Label Fraction')
plt.savefig("../part_3_results/error_rate_plot.png")


