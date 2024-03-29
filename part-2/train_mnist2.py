from pylab import * 
import matplotlib.pyplot as plt
import sys
import caffe
from caffe import layers as L, params as P
import os
caffe_root = '../../'#change accordingly
sys.path.insert(0, caffe_root + 'python')
os.chdir(caffe_root)
os.chdir('examples/part-2') #change accordingly

def define_net(lmdb, batch_size): #This function creates the model definitions
	n = caffe.NetSpec()
	n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
                             transform_param=dict(scale=1./255), ntop=2) #This Defines data layer and the source of the data
	#Following lines specify model architecture
	n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=20, weight_filler=dict(type='xavier'))
	n.pool1 = L.Pooling(n.conv1, kernel_size=2, stride=2, pool=P.Pooling.MAX)
	n.conv2 = L.Convolution(n.pool1, kernel_size=5, num_output=50, weight_filler=dict(type='xavier'))
	n.pool2 = L.Pooling(n.conv2, kernel_size=2, stride=2, pool=P.Pooling.MAX)
	n.fc1 =   L.InnerProduct(n.pool2, num_output=500, weight_filler=dict(type='xavier'))
	n.relu1 = L.ReLU(n.fc1, in_place=True)
	n.score = L.InnerProduct(n.relu1, num_output=10, weight_filler=dict(type='xavier'))
	n.loss =  L.SoftmaxWithLoss(n.score, n.label)

	return n.to_proto()

#Read command line arguments

if(len(sys.argv)<2):
	val = 8
elif((int(sys.argv[1]) == 8) | (int(sys.argv[1])==32)| (int(sys.argv[1])==128)):
	val = int(sys.argv[1])
else:
	val = 8
#Creates model files and specifies source of training and testing data

with open('../caffemodel-2/lenet_train_second_'+ str(val)+'.prototxt', 'w') as f:
	f.write(str(define_net('../lmdb_data/train_lmdb_part2_'+ str(val), 64)))

with open('../caffemodel-2/lenet_test_second.prototxt', 'w') as f:
    f.write(str(define_net('../lmdb_data/mnist_test_lmdb', 1)))


