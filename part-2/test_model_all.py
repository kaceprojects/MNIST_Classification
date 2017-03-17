import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import * 
import numpy
import sys
import caffe
import os


caffe.set_device(1)
caffe.set_mode_gpu()
class_labels = zeros(10)

test_label = numpy.loadtxt('../lmdb_data/testing_label.txt')
#Loop computes number of samples per class
for i in range(0, len(test_label)):
	class_labels[int(test_label[i])]+=1

#Values of std-dev
values = [8, 32, 128]
error_rate = []


for i in range(0, len(values)):
	net = caffe.Net('../caffemodel-2/lenet_test_second.prototxt','../caffemodel-2/lenet2_'+ str(values[i])+'_iter_1000.caffemodel', caffe.TEST)
#Calculating test error rate and classwise error rate
	class_err = zeros(10)
	total_err = 0

	for test_itr in range(0, 10000):
	        net.forward()
	        if(net.blobs['score'].data.argmax(1)!=net.blobs['label'].data): #Compares the label against the score
	                total_err +=1
	                class_err[int(net.blobs['label'].data)]+=1

	print float(total_err/100.0), '%'
	error_rate.append(float(total_err/100.0))
	print 'Classwise error-rate'
	print class_err/class_labels


print "Plotting Graph"
#plotting values
plt.scatter(values, error_rate)
plt.xlabel('Standard Deviation of Noise')
plt.ylabel('Error-Rate')
plt.title('Error-Rate as a function of Standard Deviation')
plt.savefig("../part_2_results/error_rate_plot.png")


