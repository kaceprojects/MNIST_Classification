# MNIST_Classification
This project trains a deep learning model over the MNIST database in different ways according to some criterion as specified below. 
**Part 1: Classifier on original data**
Use the predefined split of training and test sets, train a convolutional neural network classifier on the MNIST training set and test on the test set.

Feel free to limit the training to a single epoch to save time. You are also welcome to write your own code from scratch or copy pre-existing code from the web; there is no shortage of MNIST-training tutorials in the wild. However, you will be responsible for code cleanliness, efficiency and organization, so don’t copy bad code! Please do cite your source for any copied code.

After training, please answer the following questions:
What is your test set error rate?

What is the test set error rate for each class? Are some classes more challenging than others to distinguish from each other? Why?

Based only on information gathered in the first epoch of training, do you think that the model would benefit from more training time? Why?

Besides training for a longer time, what would you do to improve accuracy?



**Part 2: Added image noise**

Now add random Gaussian noise to the training set images. Set the mean to 0 and the standard deviation to 8 (given an input image range between 0 and 255). Ensure that the output image pixel image values are properly clipped between 0 and 255. Retrain the model on the noise-added images and test it on the original (noise-free) test set. Now what is your error rate?

Repeat the process for two more standard deviation values: 32 and 128. Plot the error rate as a function of noise standard deviation.

Please answer the following questions:
What are the implications of the dependence of accuracy on noise if you were to deploy a production classifier? How much noise do you think a production classifier could tolerate?

Do you think that Gaussian noise is an appropriate model for real-world noise if the characters were acquired by standard digital photography? If so, in what situations? How would you compensate for it?

Is the accuracy of certain classes affected more by image noise than others? Why?

**Part 3: Label noise**
Go back to the original, noise-free MNIST data set. Now take 5% of the training images and randomize their labels. Retrain your classifier on the label-randomized images and test on the original (non-randomized) test set.

Then repeat the process with random labels for 15% and 50% of the training images. Plot the error rate as a function of randomized label fraction.

Please answer the following questions:
How important are accurate training labels to classifier accuracy?

How would you compensate for label noise? Assume you have a large budget available but you want to use it as efficiently as possible.

How would you quantify the amount of label noise if you had a noisy data set?

If your real-world data had both image noise and label noise, which would you be more concerned about? Which is easier to compensate for?

Before I answer any of the questions above, I have included a description of the files and how to run them. 

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

**Part - 1**
This is the plot generated for training loss as well as test accuracy. 
![Alt text](https://github.com/kathachanda/MNIST_Classification/blob/master/part_1_results/learning_curve_1.png?raw=true "Training loss and testing accuracy")


Test error rate is ~ 2%
Yes, some classes are more difficult to distinguish than others. Especially 4, 7, 6 (in that order). I believe this is because many of the samples from these classes resemble samples from other classes. This may also be because the number of training samples are less for these classes.  This problem maybe mitigated by 1. Augmenting the data. 2. Training longer. 
>> Sample error rates -  0.6122449   1.23348018  1.74418605  1.18811881  3.46232179  0.89686099 2.19206681  3.0155642   1.74537988  1.58572844


Based on the plot from training over the first epoch, the training loss seems to converge at a pretty low value . However, I believe it will be worthwhile training over a few more epochs for the training loss to go even lower and stabilize further. The loss at the end of training is in the order of 2e10-3 but training for longer may be able to reduce the training loss to less than 1e10-3. 


Besides training for a longer time, various other techniques can be tried to improve accuracy. For eg. 
Change the hyperparameters, for example the learning rate, decay, momentum, etc. 
Choose a different model  - (Deeper models have been known to perform better), so one can choose a different architecture with more layers
 Augment data to increase training samples (More data is always better!)
Add more new data (expensive and not always possible)\

**Part-2**

Std-dev = 8
![Alt text](https://github.com/kathachanda/MNIST_Classification/blob/master/part_2_results/learning_curve2_8.png?raw=true "Training loss and testing accuracy")

Std-dev = 32
![Alt text](https://github.com/kathachanda/MNIST_Classification/blob/master/part_2_results/learning_curve2_32.png?raw=true "Training loss and testing accuracy")

Std-dev = 128

![Alt text](https://github.com/kathachanda/MNIST_Classification/blob/master/part_2_results/learning_curve2_128.png?raw=true "Training loss and testing accuracy")

Error- Plot
![Alt text](https://github.com/kathachanda/MNIST_Classification/blob/master/part_2_results/error_rate_plot.png?raw=true "Training loss and testing accuracy")

Gaussian noise with std deviation of 8, 32, 128 are added in the respective images.Error rate is ~ 87.15%, 92%, 92.95%.  As the standard deviation increases, the variation around the mean increases. Which means that details from the images are lost out and therefore the performance decreases as seen below. 
Noise is always a reason for poor performance of classifiers and pre-processing is performed prior to training the classifiers with those samples. A deep model comparatively is more robust and using a deep model mitigates many of the pre-processing steps that may be needed otherwise. 
However, even a deep model is not tolerant of too much noise and that can be seen by the above images that the loss does not decrease very much but levels out at a higher value.
Because the images become more noisy with higher std-dev, the plots depict that by indicating the high training loss and the lower testing accuracy. The initial parts of the testing accuracy behave erratically, because at smaller iterations, the model doesn’t learn much and therefore many samples are misclassified. 


The amount of noise that can be tolerated by a classifier is dependent on the type of images that it is classifying. In this case, the images are small and are of handwritten digits. 
Adding noise to such an image causes many of the smaller details/edges etc to be lost out. That being said, almost all real world data is noisy. 
Common distortions may be blur, noise, contrast, etc. Based on the paper in [1], blurring always reduces the performance. It maybe because blurring causes textural details to be lost. Noise in comparison also reduces the performance, but for deeper networks, the error rate is likely to drop off slower. 


Augmenting the data by a large value, can create more samples of the same class and the model can learn to recognise noisy images. Augmenting data with slight noise is a common practice. GoogleNet is deliberately trained with images having some noise. Training the model on noisy images can make it more robust to noise. 

In some cases, gaussian noise is a reasonably accurate model for images shot by digital photography. Gaussian noise commonly occurs during image acquisition, for example, sensor noise due to poor illumination. Gaussian noise in images are usually additive and independent at each pixel. This behavior has been modelled while adding image noise in the python files. In photographic films, grain is a signal-dependent noise. A gaussian model is often used as a reasonably accurate model for such kinds of noise. 


Some classes are more affected than others while classification. 
Classes - 1, 6 and 7 are more  misclassified than the rest, for stddev 8 and 32. While for stddev 128, classes 1, 6, 8, 9 are more misclassified. This may be because the training samples in these classes may not have much variation as opposed to the testing samples. 
For many samples, 1 and 7 may look similar. This may be a point of confusion too at the time of testing. Added noise makes distinguishing smaller details much more difficult thus leading to increased misclassification. 

**Part-3**

Label accuracy is extremely important while training a classifier. This is because the labels of the training test can be considered as ground truth data. If the labels are not accurate, the classifier will be mis-trained.

Here are plots with different % of labels randomized. 

5% labels randomized
![Alt text](https://github.com/kathachanda/MNIST_Classification/blob/master/part_3_results/learning_curve3_5.png?raw=true "Training loss and testing accuracy")

15% labels randomized
![Alt text](https://github.com/kathachanda/MNIST_Classification/blob/master/part_3_results/learning_curve3_15.png?raw=true "Training loss and testing accuracy")

50% labels randomized
![Alt text](https://github.com/kathachanda/MNIST_Classification/blob/master/part_3_results/learning_curve3_50.png?raw=true "Training loss and testing accuracy")

Error rate vs % of randomised labels

![Alt text](https://github.com/kathachanda/MNIST_Classification/blob/master/part_3_results/error_rate_plot.png?raw=true "Training loss and testing accuracy")

As the classifier is trained over only one epoch, I believe the classifier has not learnt at all, with randomizd labels and since the testing accuracy is visualized over just one epoch, it cannot be seen decreasing for higher percentages of noisy labels. The shoot-ups and dips of the testing accuracy is erratic and the decrease in testing accuracy can be clearly noted if trained over a larger number of epochs. 
The graph for test accuracy as a function of percentage of randomised labels **does not** clearly depict the true relationship between the two variables in this case.

The wide variation in training loss is also a result of the mis-labelling. Since the data is mis-labelled, the error computed at each step of the stochastic gradient varies wildly instead of decreasing steadily with minor variations. 


There are many ways to train a network while compensating for noisy labels. These methods are described in a number of papers. 
[2] builds a probabilistic model that captures the relations between noisy labels, clean labels, images, noise type which is solved using EM algorithm and then integrated into the CNN. 
[4] employs a similar method of building a probabilistic model that is optimized using EM and then integrated into the back-propagation algorithm.Another way can be to remove the noisy instances. However one may not always know which data is mislabelled. Some form of anomaly detection or outlier detection may be employed to detect mislabelled data. One may even employ an ensemble learning algorithm. If every classifier in the ensemble does not agree upon the same label, it may be considered mislabelled [3]. However removing noisy instances can pose the risk of a very small resultant noise free dataset which is not very scalable. 

Label noise can be quantified in a similar manner as described above. Using some algorithm similar to the concept of k-means or by performing outlier detection, one can detect labels liable to be noisy. 


Label noise is more concerning than image noise. Image noise may be compensated by deliberately training with noisy images. However most classifiers base their excellent accuracy on the assumption that the labels aren’t noisy. In reality, very large datasets in the real world are seldom labelled with 100% accuracy. In such cases, steps must be taken to compensate for this.

 


**References**

[1] Understanding How Image Quality Affects Deep Neural Networks 
https://arxiv.org/pdf/1604.04004.pdf
[2] Xiao, Tong, et al. "Learning from massive noisy labeled data for image classification." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.
[3]Frénay, Benoît, and Ata Kabán. "A comprehensive introduction to label noise." ESANN. 2014.
[4]Bekker, Alan Joseph, and Jacob Goldberger. "Training deep neural-networks based on unreliable labels." Acoustics, Speech and Signal Processing (ICASSP), 2016 IEEE International Conference on. IEEE, 2016.

**Resources for code used:**

http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/01-learning-lenet.ipynb

http://adilmoujahid.com/posts/2016/06/introduction-deep-learning-python-caffe/

MNIST parser:
https://github.com/sorki/python-mnist

















