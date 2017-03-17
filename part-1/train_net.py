import matplotlib
matplotlib.use('Agg')
import numpy
from pylab import * 
import sys
import caffe
import os
import cv2


caffe.set_device(1)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('../caffemodel-1/lenet_auto_solver.prototxt')

niter = 1000
test_interval = 100
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter / test_interval)))
output = zeros((niter, 8, 10))

# the main solver loop
#May be done by command line however manually looping lets us perform other different computations in the loop 
for it in range(niter):
    solver.step(1)  # 1 step ofSGD by Caffe
    # store the train loss
    train_loss[it] = solver.net.blobs['loss'].data
    # store the output of the first test batch
    # (starting the forward pass at conv1 to avoid loading new data)
    solver.test_nets[0].forward(start='conv1')
    output[it] = solver.test_nets[0].blobs['score'].data[:8]    
    #Running a full test on each 100th iteration and computing test accuracy
    if it % test_interval == 0:
        print 'Iteration', it, 'testing...'
        correct = 0
        for test_it in range(10000):

		solver.test_nets[0].forward()
		correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)== solver.test_nets[0].blobs['label'].data)
        test_acc[it // test_interval] = correct / 1e4

#plotting and storing the training loss and test accuracy

_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
ax2.set_title('Test Accuracy: {:.2f}'.format(test_acc[-1]))
plt.savefig("../part_1_results/learning_curve_1.png")


print(test_acc[-1])

