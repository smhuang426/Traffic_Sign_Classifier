# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/bar_chart.png "Visualization"
[image2]: ./examples/before_data_agu.png "Before data agument"
[image3]: ./examples/after_data_aug.png "After data augument"
[image4]: ./examples/ChoosenImage.png "Choosen Image"
[image5]: ./examples/KernelImage.png "Kernel Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/smhuang426/Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I wrote a drawDistrubution(dataSet_X, dataSet_Y) method to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

* As a first step, I decided to convert the images with histogram equalization, but the result of accuracy is :

with histogram equalization : ~91%
without histogram equalization : ~93%

in 16 epochs

No histogram equalization is better, but I cannot figure out why, because I thought histogram equalization is doing normalization. So I quit this idea.

* About normalization, I had compared (pixel - 128)/ 128 with pixel/ 256

pixel/ 256 : ~93% accuracy
(pixel - 128)/ 128 : ~71% accuracy

in 16 epochs

I thought because my acivation function is ReLU function, all of negative value will vanish as zero , so (pixel - 128) will cause negative value and some of infomation will be losing.

I decided to generate additional data because I find out some of class have little images.  

To add more data to the the data set, I used the scaling, shifting and rotating it because I want to made those data differnt from original data as data augment. 

Here is an example of an original image and an augmented image:

Before
![alt text][image2]

After
![alt text][image3]

The difference between the original data set and the augmented data set is scaling, shifting, rotating. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 3x3	    | 1x1 stride, VALID padding, outputs 12x12x64 |
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, VALID padding, outputs 10x10x128 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x128 				|
| Fully connected		| Input = 3200. Output = 2400 |
| Fully connected		| Input = 2400. Output = 1600 |
| Fully connected		| Input = 1600. Output = 43 |
| Softmax				| Output = 43        									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer, batch size = 128, number of epochs:21 and I used learning rate decay because I found that accuracy is shaking when it reach 90% above, so I shrink its step size per 7 epochs so that it can walk stable. Learning rate start with 0.008 . 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 96.8%
* test set accuracy of 95.5%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

Just a normal LeNet-5 

* What were some problems with the initial architecture?

It cannot reach 95%up accuracy in 21 epochs, only 93% accuracy for test set

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

Originaly, I use two 5 by 5 convolutional layer, but I thought increase deep of networking that maybe increase accuracy, so I change second convolutional layer to two 3*3 convolutional layer without zero padding, and It's working. 

* Which parameters were tuned? How were they adjusted and why?

I fine tune learning rate with learning rate decay, it will decay 16% per 7 epochs so that can converge stably.

* What are some of the important design choices and why were they chosen?

I choose x/255 as my normalization instead of ( x -128 )/128, that increase accuracy from 70% to 90%up in 16 epochs, becase I thought my activation function is relu, which will vanish negative value as zero. If I use (x-128)/128, some of information will be lose when start tringing.

If a well known architecture was chosen:
* What architecture was chosen?

I want to try YOLO.

* Why did you believe it would be relevant to the traffic sign application?

Because we need to find where is traffic sign in pratical application then we classify which sign is. And YOLO is the fastest object detection I'v met. 

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
Validation set and test set are close each other and up to 95%, so maybe do not have overfitting.

### Test a Model on New Images

#### 1. Choose 18 German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 18 German traffic signs that I found on the web:
![alt text][image4]


The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

It find feature for every kernel. 

![alt text][image5]

