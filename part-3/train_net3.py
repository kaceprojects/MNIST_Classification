import matplotlib
matplotlib.use('Agg')
import numpy
from pylab import * 
import sys
import caffe
import os
import cv2

if(len(sys.argv)<2):
	val = 5
elif((int(sys.argv[1]) == 5) | (int(sys.argv[1])==15)| (int(sys.argv[1])==50)):
	val = int(sys.argv[1])
else:
	val = 5
#print int(sys.argv[1])
print val
caffe.set_device(1)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('../caffemodel-3/lenet_auto_solver3_'+ str(val)+'.prototxt')

niter = 1000
test_interval = 100
# losses will also be stored in the log
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter / test_interval)))
output = zeros((niter, 8, 10))

# the main solver loop
for it in range(niter):
    solver.step(1)  # SGD by Caffe
    
    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    
    # store the output on the first test batch
    # (start the forward pass at conv1 to avoid loading new data)
    solver.test_nets[0].forward(start='conv1')
    output[it] = solver.test_nets[0].blobs['score'].data[:8]
    
    # run a full test every so often
    # (Caffe can also do this for us and write to a log, but we show here
    #  how to do it directly in Python, where more complicated things are easier.)
    if it % test_interval == 0:
        print 'Iteration', it, 'testing...'
        correct = 0
        for test_it in range(10000):

		solver.test_nets[0].forward()
		correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)
                           == solver.test_nets[0].blobs['label'].data)
#       		print(solver.test_nets[0].blobs['label'].data)
#		print(solver.test_nets[0].blobs['score'].data.argmax(1))
        test_acc[it // test_interval] = correct / 1e4

_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
ax2.set_title('Test Accuracy: {:.2f}'.format(test_acc[-1]))
plt.savefig("../part_3_results/learning_curve3_"+ str(val)+ ".png")


print(test_acc[-1])
#Calculating test error rate and classwise error rate 
#prints a 10 x 1 matrix with the number of wrong evaluations per class
#Must divide by total actual labels per class later on
