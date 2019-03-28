# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[test_set_histogram]: ./writeup_images/test_set_histogram.png "Test set histogram"
[train_set_histogram]: ./writeup_images/train_set_histogram.png "Train set histogram"
[valid_set_histogram]: ./writeup_images/valid_set_histogram.png "Validation set histogram"
[exampleimage1]: ./writeup_images/exampleimage1.png "Dataset images"
[german_1_sign]: ./german_signs_test/1_sign.jpg "30km/h sign"
[german_4_sign]: ./german_signs_test/4_sign.jpg "70km/h sign"
[german_13_sign]: ./german_signs_test/13_sign.jpeg "Yield sign"
[german_17_sign]: ./german_signs_test/17_sign.jpeg "No entry sign"
[german_38_sign]: ./german_signs_test/38_sign.jpg "Keep right sign"



### Data Set Summary & Exploration

In this project, I've used a german traffic sign dataset. It consisted of 43 different signs on 32x32 pixels,
with 34799 images on the training set, 4410 images on validation set 12630 on the test set.  

The availability of each sign has a drastic difference. Some more common signs have over 2000 samples on
the training set, while others have less than 200, as we can see on the following histograms:


![alt text][train_set_histogram]
![alt text][valid_set_histogram]
![alt text][test_set_histogram]

And here are some example images of the dataset:

![alt text][exampleimage1]



### Design and Test a Model Architecture

First step, I've prepared the images for the neural network. I did a lot of exploratory work on this part
like creating image transformations, HSL and HSV transforms, histogram normalized grayscale images, but
I've got the best results with RGB channels. My intuition is that the color of the signs are important 
for the determination of the sign, and thus proved a valuable information for the neural network.
However, I've transformed the range of values from 0 to 255 to -1 and 1.


#### Neural network layers

I've implemented a neural network based in a LeNet architecture, but with an additional layer before the fully connected
layers. 

My final model consisted of the following layers:

| Layer         		                 |     Description	        					| 
|:--------------------------------------:|:--------------------------------------------:| 
| Input         		                 | 32x32x3 RGB image   							| 
| Convolution 5x5 + RELU                 | 2x2 stride, VALID padding, shape 5x5x12  	|
| Convolution 5x5 + RELU                 | 1x1 stride, VALID padding, shape 5x5x24      |
| Convolution 10x10 + RELU + Max pooling | 1x1 stride,  shape 10x10x32     		    |
| Fully connected	                   | Input 1158, output 250    						|
| Fully connected		               | Input 250, output 86        				    |
| Fully connected	                   | Input 86, output 43							|



#### Model training

To train the model, I used the following parameters:

````python
EPOCHS = 30
BATCH_SIZE = 32
learning_rate = 0.0005
keep_prob = 0.8

````

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

#### Model results

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 98% 
* test set accuracy of 96%

The first architecture was a plain LeNet architecture, which was giving satisfactory results since
the beginning, but I decided to tweak and add more layers to it.

I fist added one extra internal convolutional layer. It increased the accuracy from 93% to around 94.5%.
Then, I added one more layer and removed the pooling on internal layers. This increased the accuracy from 94.5% to around 96%.

I also tried increasing the number of inputs with other image transformations, like grayscale, HSV and HSL transformations,
but it did not improve the result, which proved that the color information is actually valuable for accurately detecting
the signs.

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][german_1_sign] ![alt text][german_4_sign] ![alt text][german_13_sign] 
![alt text][german_17_sign] ![alt text][german_38_sign]


#### Results

Here are the results of the prediction:

| Image		  |     Prediction	                     | 
|:-----------:|:------------------------------------:| 
| 70 km/h     | 60 km/h                              | 
| 30 km/h     | OK! 								 |
| Yield		  | OK!                                  |					
| No entry	  | OK!					 				 |
| Keep right  | OK!      						 	 |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. I did not expect that
it would miss the 70 km/h sign, because this sign was one of the most common on the dataset. My guess is that the model
is overfitted and did not handle very well the image from the internet.

It's also worth mentioning the top 5 guesses of each sign and the confidence on each one of them.
On the sign that it misclassified, we can see that the sign is not even on the list, and it has a 
very strong confidence it was a 60 km/h sign 

**Speed limit (70km/h)**  
Guess 1: Speed limit (60km/h) -> (100%)  
Guess 2: Turn left ahead -> (0%)  
Guess 3: Bicycles crossing -> (0%)  
Guess 4: No passing -> (0%)  
Guess 5: Dangerous curve to the right -> (0%)  

**Keep right**  
Guess 1: Keep right -> (100%)  
Guess 2: Speed limit (20km/h) -> (0%)  
Guess 3: Speed limit (30km/h) -> (0%)  
Guess 4: Speed limit (50km/h) -> (0%)  
Guess 5: Speed limit (60km/h) -> (0%)  

**Speed limit (30km/h)**  
Guess 1: Speed limit (30km/h) -> (100%)  
Guess 2: Speed limit (80km/h) -> (0%)  
Guess 3: Speed limit (20km/h) -> (0%)  
Guess 4: Roundabout mandatory -> (0%)  
Guess 5: Speed limit (50km/h) -> (0%)  

**Yield**  
Guess 1: Yield -> (100%)  
Guess 2: Speed limit (50km/h) -> (0%)  
Guess 3: Speed limit (30km/h) -> (0%)  
Guess 4: Road work -> (0%)  
Guess 5: Priority road -> (0%)  

**No entry**  
Guess 1: No entry -> (100%)  
Guess 2: Dangerous curve to the left -> (0%)  
Guess 3: No passing -> (0%)  
Guess 4: Speed limit (20km/h) -> (0%)  
Guess 5: General caution -> (0%)  


### Possible improvements

One of ways we could improve this model is to further investigate more layer structures and parameters
that we could use in order to improve the neural network performance. Also, a more evenly distributed dataset,
with more samples of some signs would also be beneficial for the results.