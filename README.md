# MNIST_Classification
This project trains a deep learning model over the MNIST database in different ways according to some criterion as specified below. 
**Part 1: Classifier on original data**
Use the predefined split of training and test sets, train a convolutional neural network classifier on the MNIST training set and test on the test set.

Feel free to limit the training to a single epoch to save time. You are also welcome to write your own code from scratch or copy pre-existing code from the web; there is no shortage of MNIST-training tutorials in the wild. However, you will be responsible for code cleanliness, efficiency and organization, so don’t copy bad code! Please do cite your source for any copied code.

**Part 2: Added image noise**

Now add random Gaussian noise to the training set images. Set the mean to 0 and the standard deviation to 8 (given an input image range between 0 and 255). Ensure that the output image pixel image values are properly clipped between 0 and 255. Retrain the model on the noise-added images and test it on the original (noise-free) test set. Now what is your error rate?

Repeat the process for two more standard deviation values: 32 and 128. Plot the error rate as a function of noise standard deviation.

**Part 3: Label noise**
Go back to the original, noise-free MNIST data set. Now take 5% of the training images and randomize their labels. Retrain your classifier on the label-randomized images and test on the original (non-randomized) test set.

Then repeat the process with random labels for 15% and 50% of the training images. Plot the error rate as a function of randomized label fraction.



The code is written with python 2.7 and uses OpenCV 3. It uses Caffe as a deep learning framework.
To run, place the contents of the repository in the examples folder within caffe root. If you want to place it elsewhere, change the paths accordingly. 

All open source code libraries are used. 

Install MNIST parser from here : 
https://github.com/sorki/python-mnist

**Execute the following commands from terminal**

cd $CAFFE_ROOT

./data/mnist/get_mnist.sh

./examples/mnist/create_mnist.sh

Run the test_train_label.py file in the misc folder.

Copy and paste the generated mnist_train/test_lmdb folders into the lmdb_data folder of this repository.

To run part 1, first initialize the network by running train_mnist.py. Then train it by running train_net.py. Finally, test it on the mnist test database by running test_model_1.py

Part 2 generates 3 different models trained with 3 different training data.

To generate the data, run add_noise.py along with command line arguments of either 8, 32 or 128. 

For example, python add_noise.py 32

To initialize a particular model, run train_mnist2.py with command line arguments of either 8, 32 or 128

For example, python train_mnist2.py 32

To train it, run train_net2.py with similar command line arguments as above. 

To test it, run test_model_2.py with similar command line arguments as above. 

To compare their error rates, run test_model_all.py. 

This will output the error rate as a percentage for each model as well as the error rate per class for each model and a plot of error rate vs std-dev [you can run this as-is only if you have all 3 trained models]

Part 3 generates 3 different models trained with 3 different training data. 

To generate the data, run create_data_part3.py along with command line arguments of either 5, 15 or 50. 

For example, python create_data_part3.py 50

To initialize a particular model, run train_mnist3.py with command line arguments of either 5, 15 or 50

For example, python train_mnist3.py 50

To train it, run train_net3.py with similar command line arguments as above. 

To test it, run test_model_3.py which outputs the error rate as a percentage for each model as well as error rate per class for each model and a plot of error rate vs label randomization [you can only run this as-is if you have all 3 trained models]. Separate test files are not made for this part as the only requirement is to plot the error rate vs label randomization. 

The deep model used for digit classification here is a modification of Yann Lecun’s LeNet which is a convolutional neural network with two layers. 
LeNet was chosen because of a number of reasons. It is small enough to train quickly. It has a simple, easy-to-understand architecture. Yann LeCun originally trained the MNIST database on LeNet hence, even though it will not achieve state-of-the-art performance, a reasonably high accuracy can be expected. 

Because of limited computing resources, all nets are trained approximately for each epoch. 
For the same reasons, no data augmentation is performed. 
Number of training_samples = size of train batch * number of iterations per epoch. 
64 is chosen as a training batch size and total training samples are 60000 therefore number of iterations in 1 epoch is 935 ~ 1000. 

The accuracy layer in the original LeNet is replaced by a score layer and accuracy is manually computed. 

Test error rate is defined as (FP + FN)/(TP + TN + FP + FN), which is basically defined as all wrongly classified samples divided by all samples. 


**Resources for code used:**

http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/01-learning-lenet.ipynb

http://adilmoujahid.com/posts/2016/06/introduction-deep-learning-python-caffe/

MNIST parser:
https://github.com/sorki/python-mnist

















