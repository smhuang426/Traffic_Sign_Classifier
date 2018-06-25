# Load pickled data
import pickle
import glob
import cv2

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'train.p'
validation_file= 'valid.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

print( 'Shape of X_train:', X_train.shape )
print( 'Shape of y_train:', y_train.shape )
print( 'Shape of X_valid:', X_valid.shape )
print( 'Shape of X_test:' , X_test.shape )



### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of validation examples
n_validation = X_valid.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[1:4]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = 43

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)



### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
import numpy as np

%matplotlib inline

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() - 0.1, 1.03*height, '%s' % int(height))
        
def drawDistrubution(dataSet_X, dataSet_Y):
    ###Draw a histogram for class distribution
    train_values = np.zeros(( n_classes ))
    for idx in range(len(dataSet_X)):
        train_values[dataSet_Y[idx]] += 1

    plt.figure(figsize=(20, 4))
    rect = plt.bar(range(n_classes), train_values)
    plt.xticks(range(n_classes))
    plt.xlabel("Class number")
    plt.ylabel("Number of images")

    autolabel(rect)
    plt.show()
    
def draw_first_sign( x_data, y_data ):
    plt.figure(figsize=(10, 18.5))
    for i in range(0, n_classes):
        plt.subplot(9, 6, i+1)
        x_selected = x_data[y_data == i]
        plt.imshow(x_selected[0, :, :, :]) #draw the first image of each class
        plt.title(i)
        plt.axis('off')
    plt.show()
	
###Draw first sign from every class
draw_first_sign( X_train, y_train )

###Draw a histogram for class distribution
print('Before do data augment:')
drawDistrubution(X_train, y_train)



### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
X_trainSet = []
Y_trainSet = []

def equal_hist(img):
    #Histogram Equalization
    img2=img.copy() 
    img2[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img2[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img2[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    return img2

def sharpen_img(img):
    gb = cv2.GaussianBlur(img, (5,5), 20.0)
    return cv2.addWeighted(img, 2, gb, -1, 0)

def scale_img(img):
    img2=img.copy()
    sc_y=0.4*np.random.rand()+1.0
    sc_x=0.4*np.random.rand()+1.0
    img2=cv2.resize(img, None, fx=sc_x, fy=sc_y, interpolation = cv2.INTER_CUBIC)
    orgY = int(( sc_y - 1 ) * np.random.rand() * img.shape[0] )
    orgX = int(( sc_x - 1 ) * np.random.rand() * img.shape[1] )
    return img2[ orgY: orgY + img.shape[0], orgX: orgX + img.shape[1], :]

def rotate_img(img):
    c_x,c_y = int(img.shape[0]/2), int(img.shape[1]/2)
    ang = 40.0*np.random.rand()-20
    Mat = cv2.getRotationMatrix2D((c_x, c_y), ang, 1.0)
    return cv2.warpAffine(img, Mat, img.shape[:2])

	
####################
### Data augment ###
####################
for idx in range(len(X_train)):
    X_trainSet.append( X_train[idx] / 256 )    # Deivide 256 because it's normalization
    Y_trainSet.append( y_train[idx] )
    ###I choose some classes to augment data
    if y_train[idx] == 0 or y_train[idx] == 16 or \
        y_train[idx] == 20 or y_train[idx] == 21 or \
        y_train[idx] == 24 or y_train[idx] == 27 or \
        y_train[idx] == 29 or y_train[idx] == 32 or \
        y_train[idx] == 34 or y_train[idx] == 40 or \
        y_train[idx] == 41 or y_train[idx] == 42 or \
        y_train[idx] == 16 or y_train[idx] == 22 or \
        y_train[idx] == 28 :
            
        X_trainSet.append( scale_img(X_train[idx]) / 256 )   # Deivide 256 because it's normalization
        X_trainSet.append( rotate_img(X_train[idx]) / 256 )  # Deivide 256 because it's normalization
        Y_trainSet.append( y_train[idx] )
        Y_trainSet.append( y_train[idx] )

### I did sharpened and histogram equalization for all of data, but it's not obvious improvment.
#for m in range(n_train):
#    X_train[m] = sharpen_img(X_train[m])
#    X_train[m] = equal_hist(X_train[m])
#    X_train[m] = scale_img(X_train[m])

#for m in range(n_validation):
#    X_valid[m] = sharpen_img(X_valid[m])
#    X_valid[m] = equal_hist(X_valid[m])

#for m in range(n_test):
#    X_test[m] = sharpen_img(X_test[m])
#    X_test[m] = equal_hist(X_test[m])

###Draw first sign from every class
draw_first_sign( X_valid, y_valid )

print('After do data augment:')
drawDistrubution(X_trainSet, Y_trainSet)

### Do normalize with all off data
#X_trainSet = ( X_trainSet ) / 256
X_validSet = ( X_valid ) / 256
X_testSet  = ( X_test ) / 256

n_train = len(X_trainSet)
print('new number of traning set:',n_train)



fromfrom  tensorflow.contrib.layerstensorf  import flatten
import tensorflow as tf

####################
#Model Architecture#
####################
### Define architecture here.
def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x32.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 32), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(32))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x32. Output = 14x14x32.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 12x12x64.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 32, 64), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(64))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # Activation.
    conv2 = tf.nn.relu(conv2)
    
    # Layer 3: Convolutional. Output = 10x10x128.
    conv3_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 128), mean = mu, stddev = sigma))
    conv3_b = tf.Variable(tf.zeros(128))
    conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='VALID') + conv3_b
    
    # Activation.
    #conv2 = tf.nn.relu(conv2)
    conv3 = tf.nn.relu(conv3)

    # Pooling Input = 10x10x128. Output = 5x5x128.
    #conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x128. Output = 3200.
    #fc0   = flatten(conv2)
    fc0   = flatten(conv3)
    
    # Layer 3: Fully Connected. Input = 3200. Output = 2400.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(3200, 2400), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(2400))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # Activation.
    fc1    = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 2400. Output = 1600.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(2400, 1600), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(1600))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # Activation.
    fc2    = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 1600. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(1600, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits, conv1, conv2

###Create evaluate method, refered to LeNet-Lab
def evaluate(X_data, y_data, sess):
    total_accuracy = 0
    num_examples = len(X_data)
    
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(evaluating, feed_dict={input_x: batch_x, expected_y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
        
    return total_accuracy / num_examples


def analyz_err(X_data, y_data, sess):

    analysis = np.zeros((n_classes))
    values   = np.zeros((n_classes))
    
    predict_ret, onehot_idxs = sess.run(analyzing, feed_dict={input_x: X_data, expected_y: y_data})
    
    for idx in range(len(y_data)):
        values[y_data[idx]] += 1
    
    for idx in range(len(predict_ret)):
        if predict_ret[idx] == False:
            analysis[onehot_idxs[idx]] += 1
            
    print(len(predict_ret), sum(analysis))
        
    return analysis, values


#Import shuffle
from sklearn.utils import shuffle

###Setting Parameters
EPOCHS     = 21
BATCH_SIZE = 128

###Setting learning rate
rate = 0.0008

###Declare input_x: CNN's input, expected_y: expected output / one-hot label
input_x    = tf.placeholder(tf.float32, (None, 32, 32, 3))
expected_y = tf.placeholder(tf.int32, (None))
one_hot_y  = tf.one_hot(expected_y, n_classes)

lrning_rate = tf.placeholder(tf.float32)

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
logits, cov1, cov2 = LeNet(input_x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss          = tf.reduce_mean(cross_entropy)
training      = tf.train.AdamOptimizer(learning_rate = lrning_rate).minimize(loss)

###Evaluate accuracy
prediction    = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
evaluating    = tf.reduce_mean(tf.cast(prediction, tf.float32))

analyzing = prediction, tf.argmax(one_hot_y, 1)

###Create a saver to store training result
saver = tf.train.Saver()

#Get session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print('Start Training!!\n')

for i in range(EPOCHS):
    
    #learning rate decay
    if (i % 7) == 6:
        rate *= 0.82
    
    X_trainSet, Y_trainSet = shuffle(X_trainSet, Y_trainSet)
    for offset in range(0, n_train, BATCH_SIZE):
        end = offset + BATCH_SIZE
        batch_x, batch_y = X_trainSet[offset:end], Y_trainSet[offset:end]
        sess.run(training, feed_dict={input_x: batch_x, expected_y: batch_y, lrning_rate:rate})
            
    validation_accuracy = evaluate(X_validSet, y_valid, sess)
    print("EPOCH {} ...".format(i+1))
    print("Validation Accuracy = {:.3f}".format(validation_accuracy))
    print()
    
    error_analyze, sum_value = analyz_err(X_validSet, y_valid, sess)
    for idx in range(n_classes):
        print(idx,":",error_analyze[idx]," total:",sum_value[idx],"error rate:",error_analyze[idx]/sum_value[idx]*100)
    print()
    print()

saver.save(sess, './lenet')
print("Model saved")

###Read the trained model
saver.restore(sess, tf.train.latest_checkpoint('.'))

test_accuracy = evaluate(X_testSet, y_test, sess)
print("Test Accuracy = {:.3f}".format(test_accuracy))
print()

error_analyze, sum_value = analyz_err(X_testSet, y_test, sess)
for idx in range(n_classes):
        print(idx,":",error_analyze[idx]," total:",sum_value[idx],"error rate:",error_analyze[idx]/sum_value[idx]*100)
		

############################
#Test a Model on New Images#
############################		
import csv

### Load the images and plot them here.
### Feel free to use as many code cells as needed.
test_img_paths = [img_path for img_path in glob.glob("Test_Image/*")]
X_test_data    = np.uint8(np.zeros((len(test_img_paths),32,32,3)))
Y_test_data    = []

### Get the label for X_test_data
idx = 0
with open('GT-final_test.csv', 'rt') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=';')
    for row in reader:
        if idx < len(test_img_paths):
            if ('Test_Image/'+row['Filename']) == test_img_paths[idx]:
                Y_test_data.append(int(row['ClassId']))
                idx += 1;
        else:
            break

###Draw sign from X_test_data
plt.figure(figsize=(10, 18.5))

print('The Ground Truth:\n')

for i in range(len(test_img_paths)):
    img = cv2.imread(test_img_paths[i], cv2.IMREAD_COLOR)
    img = cv2.resize(img, None, fx=32.0/img.shape[1], fy=32.0/img.shape[0] , interpolation = cv2.INTER_CUBIC)
    b,g,r          = cv2.split(img)
    img            = cv2.merge([r,g,b])
    X_test_data[i] = img
    
    plt.subplot(9, 6, i+1)
    plt.imshow(X_test_data[i]) #draw the first image of each class
    plt.title('Class:'+str(Y_test_data[i]))
    plt.axis('off')

plt.show()


### Run the predictions here and use the model to output the prediction for each image.
predict_ressult = []

predict = sess.run(tf.argmax(logits, 1), feed_dict={input_x: X_test_data})

for idx in range(len(predict)):
    predict_ressult.append( predict[idx] )

print('Prediction:\n', predict_ressult)
print('Truth:\n', Y_test_data)

### Calculate the accuracy for these 5 new images. 
test_accuracy = evaluate(X_test_data, Y_test_data, sess)
print("Test Accuracy = {:.3f}".format(test_accuracy*100),'%')
print()

### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
signs_class=[]
with open('signnames.csv', 'rt') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',')
    for row in reader:
        signs_class.append((row['SignName']))

softmax = sess.run(tf.nn.softmax(logits), feed_dict={input_x: X_test_data})    
top_5   = tf.nn.top_k(softmax, k=5)

top_5_result = sess.run(top_5)

self_test_len = len(X_test_data)
plt.figure(figsize=(16, 27))
for i in range(self_test_len):
    plt.subplot(self_test_len, 2, 2*i+1)
    plt.imshow(X_test_data[i]) 
    plt.title(signs_class[Y_test_data[i]])
    plt.axis('off')
    plt.subplot(self_test_len, 2, 2*i+2)
    plt.barh(np.arange(1, 6, 1), top_5_result.values[i, :])
    labs=[signs_class[j] for j in top_5_result.indices[i]]
    plt.yticks(np.arange(1, 6, 1), labs)
plt.show()


### Visualize network's feature maps.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={input_x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(8,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")


###Implemt outputFeatureMap to see what's going on.
inputX = X_test_data[10].reshape((1, 32, 32, 3))

plt.imshow(inputX[0])
plt.show()

outputFeatureMap(inputX, cov1, activation_min=-1, activation_max=-1 ,plt_num=1)