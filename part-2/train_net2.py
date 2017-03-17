import matplotlib
matplotlib.use('Agg')
import numpy
from pylab import * 
import sys
import caffe
import os

# Read command line arguments
if(len(sys.argv)<2):
	val = 8
elif((int(sys.argv[1]) == 8) | (int(sys.argv[1])==32)| (int(sys.argv[1])==128)):
	val = int(sys.argv[1])
else:
	val = 8



caffe.set_device(1) # Change accordingly
caffe.set_mode_gpu()
solver = caffe.SGDSolver('../caffemodel-2/lenet_auto_solver2_'+ str(val)+'.prototxt')

niter = 1000
test_interval = 100
train_loss = zeros(niter)
test_acc = zeros(int(np.ceil(niter / test_interval)))
output = zeros((niter, 8, 10))

# the main solver loop, Can be done via command line but doing it via code allows us mto perform other different computations while training
for it in range(niter):
    solver.step(1)  # 1 step of SGD by Caffe
    train_loss[it] = solver.net.blobs['loss'].data    
    # storing the output of the first test batch
    # starting the forward pass at conv1 to avoid loading new data 
    solver.test_nets[0].forward(start='conv1')
    output[it] = solver.test_nets[0].blobs['score'].data[:8]   
    # Running a full test at 100th iteration
    if it % test_interval == 0:
        print 'Iteration', it, 'testing...'
        correct = 0
        for test_it in range(10000):

		solver.test_nets[0].forward()
		correct += sum(solver.test_nets[0].blobs['score'].data.argmax(1)== solver.test_nets[0].blobs['label'].data)
        test_acc[it // test_interval] = correct / 1e4

# Plot training loss and testing accuracy

_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.plot(arange(niter), train_loss)
ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('test accuracy')
ax2.set_title('Test Accuracy: {:.2f}'.format(test_acc[-1]))
plt.savefig("../part_2_results/learning_curve2_"+ str(val)+".png")


print(test_acc[-1])

