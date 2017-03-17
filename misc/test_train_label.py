from mnist import MNIST
import numpy

# Saves training labels

mndata = MNIST('../training-files')
images, labels = mndata.load_training()
filename = '../lmdb_data/training_label.txt'
numpy.savetxt(filename, labels, delimiter = ',')

#Saves testing labels

images, labels = mndata.load_testing()
filename = '../lmdb_data/testing_label.txt'
numpy.savetxt(filename, labels, delimiter = ',')
