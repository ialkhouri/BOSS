import tensorflow as tf
from keras.utils import np_utils
import glob
#import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
#import PIL
from tensorflow.keras import layers
import time

#from skimage.measure import compare_ssim
from skimage.measure import compare_ssim


import pydot
import graphviz

from numpy import linalg as LA

from IPython import display

from tensorflow.keras import Input, Model

from keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import pickle
import pandas as pd
import random
import cv2

#########################################################################################################################################
############################### relu_scaler_  ######
#########################################################################################################################################
def relu_scaler_(x):
    '''

    :param x: scaler
    :return: y=relu(x)
    '''
    y=0
    if x >= 0:
        y=x
    else:
        y=0
    return y
#########################################################################################################################################


#########################################################################################################################################
############################### SSIM fucntion  ######
#########################################################################################################################################
def SSIM_index(imageA, imageB):

    imageA = imageA.reshape(32, 32)
    imageB = imageB.reshape(32, 32)

    # rho_inf = LA.norm(input_image.reshape(784, 1) - X_test_pert[idx].reshape(784, 1) , np.inf)
    (D_s, diff) = compare_ssim(imageA, imageB, full=True)
    return D_s
#########################################################################################################################################



# calculate the kl divergence

##########################################################################################
############################### jensen shannon divergence fucntion -  ######
##########################################################################################
"""
it is a normalized and stable version of the KL divergence and return values between [0,1] where 0 is two identical distributions
"""

from scipy.spatial.distance import jensenshannon

from math import log2
def D_JS_PMFs(p, q):
    # D_JS_PMFs(p,q) = D_JS_PMFs(q,p)
    return jensenshannon(p, q, base=2)

############################################################################################

# from math import log2
# # calculate the kl divergence
# def kl_divergence(p, q):
# 	return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))
#
# # calculate the kl divergence



#################################################################
########################################## load data set
#################################################################

"""
Download GTSRB data
"""

X_train, y_train = train_data['features'], train_data['labels']
X_val, y_val = val_data['features'], val_data['labels']
X_test, y_test = test_data['features'], test_data['labels']


#################################################################
########################################## process dataset
#################################################################

def grayscale(img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  return img


def gray2BGR(img):
  img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
  return img

def equalize(img):
  img = cv2.equalizeHist(img)
  return img


def preprocessing(img):
  img = grayscale(img)
  img = equalize(img)
  #normalize the images, i.e. convert the pixel values to fit btwn 0 and 1
  img = img/255
  return img

y_train = to_categorical(y_train, 43)
y_val = to_categorical(y_val, 43)
y_test = to_categorical(y_test, 43)

X_train = np.array(list(map(preprocessing, X_train)))
X_val = np.array(list(map(preprocessing, X_val)))
X_test = np.array(list(map(preprocessing, X_test)))


X_train = X_train.reshape(34799, 32, 32, 1)
X_val = X_val.reshape(4410, 32, 32, 1)
X_test = X_test.reshape(12630, 32, 32, 1)

#################################################################
########################################## load model
#################################################################

"""
The source of this trained model is:
https://github.com/ItsCosmas/Traffic-Sign-Classification/blob/master/Traffic_Sign_Classification.ipynb
"""

trained_model = keras.models.load_model("my_model_GTSRB.h5")

#################################################################
########################################## test model
#################################################################
# # CA = 97.54
# results = trained_model.evaluate(X_test, y_test)
# print("test loss, test acc:", results)


######## freeze trained_model
for layer in trained_model.layers:
    layer.trainable = False


########################################################################################
###########################################################################
#################################### BUILDING THE gen model g(z,\phi)
###########################################################################





gen_NN = tf.keras.Sequential()
initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=102)
## ADDING THE GEN MODEL layers that will be trained

layer = layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(100,), name='dense_gen', kernel_initializer=initializer)
layer.trainable=True
gen_NN.add(layer)

layer = layers.BatchNormalization()
layer.trainable=True
gen_NN.add(layer)

layer = layers.LeakyReLU()
layer.trainable=True
gen_NN.add(layer)

layer = layers.Reshape((8, 8, 256))
layer.trainable=True
gen_NN.add(layer)
#assert combined_NN.output_shape == (None, 7, 7, 256)

layer = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=initializer)
layer.trainable=True
gen_NN.add(layer)
#assert gen_NN.output_shape == (None, 14, 14, 64)

layer = layers.BatchNormalization()
layer.trainable=True
gen_NN.add(layer)

layer = layers.LeakyReLU()
layer.trainable=True
gen_NN.add(layer)

layer = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=initializer)
layer.trainable=True
gen_NN.add(layer)
#assert gen_NN.output_shape == (None, 14, 14, 64)

layer = layers.BatchNormalization()
layer.trainable=True
gen_NN.add(layer)

layer = layers.LeakyReLU()
layer.trainable=True
gen_NN.add(layer)

layer = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='sigmoid', kernel_initializer=initializer)
layer.trainable=True
gen_NN.add(layer)


# # below is added for the 1D modification
# layer = layers.Reshape((784, 1, 1))
# layer.trainable=True
# gen_NN.add(layer)




#########################################################################################################################################
############################### this is NOT sequentiail traning (traning two loss fucntions from different heads of the NN) ######
#########################################################################################################################################

### define X_d as the desired
X_desired = X_test[93]


# these two need to be have the same values as of now since X_train is the same for both
batch_size_gen = 80
batch_size_2 = 80


#################### training steps and stopping thresholds
delta_s  = 0.25

delta_js = 0.25
delta_ssim = 0.85

delta_c = 0.25

traning_steps = 20



############################################################
###  automated desired for confidence reduction y_d (desired PMF)
################################################################
number_of_classes  = 43
# target_class       =  test_labels[21]
# desired_confidence =  0.5
#
# #1 code the confidence only
# desired_PMF_confidence = np.zeros(shape=(1,number_of_classes))
# for i in range(number_of_classes):
#     if i == target_class:
#         desired_PMF_confidence[:,i] = desired_confidence
#     else:
#         desired_PMF_confidence[:,i] = (1-desired_confidence) / (number_of_classes-1)
#
#
# #2 code the decision boundary examples between class i and class j
# desired_PMF_boundary = np.zeros(shape=(1,number_of_classes))
# class_i = 7
# class_j = 6
# #below is the values of PMF[i] and PMF[j] (i.e. maximum of 0.5)
# desired_confidence_boundary = 0.5
# for i in range(number_of_classes):
#     if i == class_i or i == class_j:
#         desired_PMF_boundary[:,i] = desired_confidence_boundary
#     else:
#         desired_PMF_boundary[:,i] = (1-(2*desired_confidence_boundary)) / (number_of_classes-2)


################################################################
##### X_train is the same for both gen and combined models #####
################################################################

# build x_train as some random input and y_train to be the desired image

# X_train is the same as z in the paper

# create one vector and repeat
X_train_one = tf.random.uniform(shape=[1,100], minval=0., maxval=1., seed=101)
X_train_one_np = X_train_one.numpy()
X_train_np = np.zeros(shape=(batch_size_gen,100))
for i in range(batch_size_gen):
    X_train_np[i,:] = X_train_one_np
X_train = tf.convert_to_tensor(X_train_np, dtype=tf.float32)
X_val_np = X_train_one_np
X_val = tf.convert_to_tensor(X_val_np, dtype=tf.float32)


############################################################
### Y_train_gen for the gen model (whcih is the image)
################################################################

# below is for the 2D image
Y_train_np_gen = np.zeros(shape=(batch_size_gen,32,32,1))
Y_val_gen = X_desired.reshape(1,32,32,1)
for i in range(batch_size_gen):
    Y_train_np_gen[i,:,:,:] = Y_val_gen.reshape(32,32,1)
# convert Y_train to tf eager tensor
Y_train_gen = tf.convert_to_tensor(Y_train_np_gen, dtype=tf.float32)


# # below is for the 1D image
# Y_train_np_gen = np.zeros(shape=(batch_size_gen,784,1,1))
# Y_val_gen = X_desired.reshape(1,784,1,1)
# for i in range(batch_size_gen):
#     Y_train_np_gen[i,:,:,:] = Y_val_gen.reshape(784,1,1)
# # convert Y_train to tf eager tensor
# Y_train_gen = tf.convert_to_tensor(Y_train_np_gen, dtype=tf.float32)





############################################################
### Y_train_combined is the y_d (desired PMF)
################################################################

##this is for unifrom distribution
# Y_train_combined = 0.1*np.ones(shape=(batch_size_2,10))
# Y_val_combined = 0.1*np.ones(shape=(1,10))

###for targeted, we need to change Y_train and Y_val:
###let the target lbl be 0, then
Y_train_combined = np.zeros(shape=(batch_size_2,number_of_classes))

Y_val_combined = np.zeros(shape=(number_of_classes))
Y_val_combined[4] = 0.5 # speed 70
Y_val_combined[13] = 0.5 # yield sign
#Y_val_combined = np.array([[1,0,0,0,0,0,0,0,0,0]])

for i in range(batch_size_2):
    Y_train_combined[i,:] = Y_val_combined

#Y_desired = Y_val_combined[0]
Y_desired = Y_val_combined



print('break')


####################################################################################################################
### defining the combined model such that its the concatenation of g, then f ==> this is defing model h in the paper
#############################################################################################################################

input = Input(shape=100)

### Calling the gen model and defining the first output (the out of model g)
#this is for the first gen

x = gen_NN.layers[0](input)
for lay in range(len(gen_NN.layers) - 1):
    layer = gen_NN.layers[lay+1]
    layer.trainable = True
    x = layer(x)
out_1 = x


x_2 = trained_model.layers[0](x)
for lay in range(len(trained_model.layers) - 1):
    layer = trained_model.layers[lay + 1]
    layer.trainable = False
    x_2 = layer(x_2)
out_2 = x_2



### defining the model: this is h(z,\psi)
combined_NN = Model(input, [out_1, out_2])

### defning the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.025)

loss_1 = tf.keras.losses.MeanSquaredError(name='LOSS_1')

loss_2 = tf.keras.losses.CategoricalCrossentropy(name='LOSS_2', from_logits=False,label_smoothing=0)

#combined_NN.fit(X_train,[Y_train_gen, Y_train_combined], epochs=10, batch_size=32, validation_data=(X_val, [Y_val_gen,Y_val_combined]), verbose=1)

dynamic_weights_selection = True
# initial losses functions weights
lambda_gen = 1
lambda_pmf = 0.01

############# trainING LOOP
for i in range(traning_steps):


    combined_NN.compile(optimizer=optimizer, loss=[loss_1, loss_2], loss_weights=[lambda_gen, lambda_pmf])
    # for lay in range(18):
    #     if lay >= 12:
    #         layer = combined_NN.layers[lay]
    #         layer.trainable = False


    # traning
    combined_NN.fit(X_train,[Y_train_gen, Y_train_combined], epochs=1, batch_size=1, validation_data=(X_val, [Y_val_gen,Y_val_combined.reshape(1,number_of_classes)]), verbose=0 )
    #combined_NN.train_on_batch(X_train, [Y_train_gen, Y_train_combined])
    # fake image at step i ==> this is X in the paper and X_val is z in the paper
    fake_image = combined_NN(X_val)[0].numpy().reshape(32, 32)


    # output probabilities at step i ==> this is J in the paper


    trained_model = keras.models.load_model("my_model_GTSRB.h5")
    output_vector_probabilities = trained_model(fake_image.reshape(1, 32, 32, 1)).numpy()[0]

    #output_vector_probabilities = combined_NN(X_val)[1].numpy().reshape(10,)

    # D_2 distance between real image and fake image at step i==> this is equation (9)
    D_2_s = LA.norm(X_desired.reshape(1024,) - fake_image.reshape(1024,),          2)
    # SSIM distance between real image and fake image at step i ==>
    D_ssim_images = SSIM_index(X_desired, fake_image)
    # D_2 distance between desired PMF and the PMF returned by the fake image ==> this is equation (3)
    D_2 = LA.norm(output_vector_probabilities-Y_desired,                  2       )
    # D_JS: JS divergence distance between desired and actual PMFs (it uses KL divergence)
    D_JS = D_JS_PMFs(output_vector_probabilities, Y_desired)


    ### THE STOPPING EXIT CRITERIA
    if D_ssim_images >= delta_ssim  and D_JS <= delta_js:
        print('BREAKING FOR IS USED with Distance SSIM = ', D_ssim_images, ' and D_JS = ', D_JS)
        break

    ### logger:
    print('training step = ', i, '; image SSIM = ', D_ssim_images, ' ; PMF_JS_Distance = ', D_JS, ' ; current loss weights = ', lambda_gen,' , ', lambda_pmf )

    ##### dynamic weight selection option in training
    if dynamic_weights_selection is True:
        lambda_gen = relu_scaler_(lambda_gen       -   0.01 * 1    * ((D_ssim_images/delta_ssim)) * np.sign((D_ssim_images/delta_ssim)-1))
        lambda_pmf = relu_scaler_(lambda_pmf       -   0.01 * 0.02 * ((delta_js/D_JS))            * np.sign((delta_js/D_JS           )-1))
    else:
        lambda_gen = 1
        lambda_pmf = 0.01


    ### SAVE THE DISTNCE AND PERTURBED IMAGE SO AS TO TAKE THE MINIMUM AT THE END OF THE TRAINING STEP (THIS IS TO OVER COME OVERFITTING DURING TRAINING)



fake_image = combined_NN(X_val)[0].numpy().reshape(32,32)

### below is the same thing (just for sanity check)
trained_model = keras.models.load_model("my_model_GTSRB.h5")
output_vector_probabilities   = trained_model(fake_image.reshape(1,32,32,1)).numpy()[0]
#output_vector_probabilities_2 = combined_NN(X_val)[1].numpy().reshape(10,)
# this is to make sure that above vectr are identical
#print('This vector MUST be zero',output_vector_probabilities-output_vector_probabilities_2)
#print(output_vector_probabilities_2)

#X_desired = X_test[93]
real_image = X_desired.reshape(32,32)



def bgr2rbg(img):
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img

real_image = gray2BGR(np.float32(X_desired).reshape(32,32))
real_image = bgr2rbg(real_image)
fake_image = gray2BGR(fake_image)
fake_image = bgr2rbg(fake_image)

#real_image_rgb = cv2.cvtColor(real_image, cv2.COLOR_RGB2GRAY)


plt.figure()
plt.subplot(2,2,1)
plt.title('Desired example')
#plt.imshow(real_image,cmap='gray',vmin=0, vmax=1)
plt.imshow(real_image)
plt.colorbar()
plt.axis('off')
plt.subplot(2,2,2)
plt.title('Generated example')
#plt.imshow(fake_image,cmap='gray',vmin=0, vmax=1)
plt.imshow(fake_image)
plt.colorbar()
plt.axis('off')
plt.subplot(2,2,4)
plt.title('Generated example PMF')
plt.stem(output_vector_probabilities)
plt.ylim(top=1.2)
plt.subplot(2,2,3)
plt.title('Desired PMF')
plt.stem(Y_val_combined)
plt.ylim(top=1.2)



print('break')



