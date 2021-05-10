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
from keras.datasets import cifar10
from keras.layers import GlobalAveragePooling2D, Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation

#from skimage.measure import compare_ssim
from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity


import pydot
import graphviz

from numpy import linalg as LA

from IPython import display

from tensorflow.keras import Input, Model

from keras.models import load_model

#from tensorflow.keras.models import load_model


import tensorflow
import keras



#########################################################################################################################################
############################### SSIM fucntion  ######
#########################################################################################################################################
def SSIM_index(imageA, imageB):

    imageA = imageA.reshape(32, 32, 3)
    imageB = imageB.reshape(32, 32, 3)

    # rho_inf = LA.norm(input_image.reshape(784, 1) - X_test_pert[idx].reshape(784, 1) , np.inf)
    (D_s, diff) = compare_ssim(imageA, imageB, full=True)
    return D_s
#########################################################################################################################################


#########################################################################################################################################
############################### relu_scaler_Ismail  ######
#########################################################################################################################################
def relu_scaler_Ismail(x):
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



# ################################################### below is the MNIST digits
#
# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
# # reshape data to fit model
# X_train = train_images.reshape(train_images.shape[0], 32,32,3)
# X_test = test_images.reshape(test_images.shape[0], 32,32,3)
# X_train, X_test = X_train/255, X_test/255
# # normalization:
# train_images = train_images / 255
# test_images = test_images / 255
# print("")
#
# y_train = np_utils.to_categorical(train_labels,10)
# y_test = np_utils.to_categorical(test_labels,10)
#
# X_test = X_test.astype(np.float32)
#
# ####################################################################################

################################ some dataset - MNIST fashion

# download mnist data

#(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

# split into train and test sets

'''
# reshape data to fit model
X_train = train_images.reshape(train_images.shape[0], 32,32,3)
X_test = test_images.reshape(test_images.shape[0], 32,32,3)
X_train, X_test = X_train/255, X_test/255
# normalization:
train_images = train_images / 255
test_images = test_images / 255
'''
# ###############################################

## load the cifar10 data set
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
# reshape data to fit model
X_train = train_images.reshape(train_images.shape[0], 32, 32, 3)
X_test = test_images.reshape(test_images.shape[0], 32, 32, 3)
X_train, X_test = X_train/255, X_test/255
# normalization:
train_images = train_images / 255
test_images = test_images / 255

y_test = np_utils.to_categorical(test_labels,10)

## get a trained model (such as the MNIST didgits)

#trained_model = load_model("MNIST_digits__avgPool_dense_softmax_together_model.h5") # input is \in [0,1]
#trained_model = load_model("MNIST_digits_trained_model_3.h5") # input is \in [0,1]
#trained_model = load_model("MNIST_digits_trained_model_2.h5") # input is \in [-1,1]

#model = load_model("MNIST_digits_trained_model_3.h5") # input is \in [0,1]
#results = trained_model.evaluate(X_test, y_test)
#print("test loss, test acc:", results)

#trained_model = load_model("MNIST_digits_trained_model_3.h5")
#trained_model = load_model("MNIST_digits_trained_model_gan_like.h5")




########################################################################################

## using CIFAR-10 model

# trained_model = load_model("cifar10-resnet20-30abc31d.pth")
# trained_model = load_model("augmented_best_model.h5", compile=False)
#
# trained_model = tensorflow.keras.models.load_model("/home/user/Desktop/ISMAIL/SSLTL/SSLTL_project/RL_adv_attacks_LP/augmented_best_model.h5")

#model = cifar10vgg()

#trained_model = tensorflow.keras.models.load_model("cifar10vgg.h5")

#trained_model = load_model("cifar10vgg.h5", compile=True)

#trained_model_1 = tensorflow.keras.models.load_model("MNIST_digits_trained_model_4.h5")

trained_model = load_model("augmented_best_model.h5")


#
#
# results = trained_model.evaluate(X_test, y_test)
# print("test loss, test acc:", results)


for layer in trained_model.layers:
    layer.trainable = False

#trained_model = load_model("best_model_improved.h5")
###########################################################################
#################################### BUILDING THE gen model g(z,\phi)
###########################################################################

trained_model.summary()


# fixin the initial weights:
initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=102)
#initializer = tf.keras.initializers.Ones()

gen_NN = tf.keras.Sequential()

## ADDING THE GEN MODEL layers that will be trained

layer = layers.Dense(8*8*256, use_bias=False, input_shape=(100,), kernel_initializer=initializer)
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

layer = layers.Conv2DTranspose(64, (5,5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=initializer)
layer.trainable=True
gen_NN.add(layer)
#assert gen_NN.output_shape == (None, 14, 14, 64)

layer = layers.BatchNormalization()
layer.trainable=True
gen_NN.add(layer)

layer = layers.LeakyReLU()
layer.trainable=True
gen_NN.add(layer)

layer = layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='sigmoid', kernel_initializer=initializer)
layer.trainable=True
gen_NN.add(layer)

gen_NN.summary()

#########################################################################################################################################
############################### this is NOT sequentiail traning (traning two loss fucntions from different heads of the NN) ######
#########################################################################################################################################

### define X_d as the desired
X_desired = X_test[82]
print('length x_desired list: ',len(X_desired))


# these two need to be have the same values as of now since X_train is the same for both
batch_size_gen = 80
batch_size_2 = 80


#################### training steps and stopping criteria
delta_s  = 0.25

delta_js = 0.20
delta_ssim = 0.85

delta_c = 0.25

traning_steps = 20

################################################################
##### X_train is the same for both gen and combined models #####
################################################################

# build x_train as some random input and y_train to be the desired image

# X_train is the same as z in the paper

# create one vector and repeat
X_train_one = tf.random.uniform(shape=[1,100], minval=0., maxval=1., seed=103)
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
Y_train_np_gen = np.zeros(shape=(batch_size_gen,32,32,3))
Y_val_gen = X_desired.reshape(1,32,32,3)
for i in range(batch_size_gen):
    Y_train_np_gen[i,:,:,:] = Y_val_gen.reshape(32,32,3)
# convert Y_train to tf eager tensor
Y_train_gen = tf.convert_to_tensor(Y_train_np_gen, dtype=tf.float32)


############################################################
### Y_train_combined is the y_d (desired PMF)
################################################################

##this is for unifrom distribution
# Y_train_combined = 0.1*np.ones(shape=(batch_size_2,10))
# Y_val_combined = 0.1*np.ones(shape=(1,10))

###for targeted, we need to change Y_train and Y_val:
###let the target lbl be 0, then
Y_train_combined = np.zeros(shape=(batch_size_2,10))
#Y_val_combined = np.array([[0,0,0,0,0,0,1,0,0,0]])
#Y_val_combined = np.array([[0.25,0.25,0,0,0,0,0,0,0.25,0.25]])
#Y_val_combined = np.array([[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]])

Y_val_combined = np.array([[0,0,0,0,0,0,0.5,0.5,0,0]])

for i in range(batch_size_2):
    Y_train_combined[i,:] = Y_val_combined

Y_desired = Y_val_combined[0]



print('break')


###################################################################
##add flatten and dense layers to trained_model?
################################################################
#trained_model.add(GlobalAveragePooling2D())
#trained_model.add(Activation('softmax'))

####################################################################################################################
### defining the combined model such that its the concatenation of g, then f ==> this is defing model h in the paper
#############################################################################################################################

input = Input(shape=100)

### Calling the gen model and defining the first output (the out of model g)
x = gen_NN.layers[0](input)
for lay in range(10):
    layer = gen_NN.layers[lay+1]
    layer.trainable = True
    x = layer(x)
out_1 = x

print(x)

#### Calling the trained model and defining the 2nd output (the output of f or h)

x_2 = trained_model.layers[0](x)
print(len(trained_model.layers))
for lay in range(len(trained_model.layers)-1):
    layer = trained_model.layers[lay + 1]
    layer.trainable = False
    x_2 = layer(x_2)
out_2 = x_2

### defining the model: this is h(z,\psi)
combined_NN = Model(input, [out_1, out_2])

### defning the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.025)

### define the loss functions for each head
#

#combined_NN.summary()


dynamic_weights_selection = True
# initial losses functions weights
lambda_gen = 1
lambda_pmf = 0.001



############# trainING LOOP
for i in range(traning_steps):
    # compile at every step so as to update the loss functions weights
    combined_NN.compile(optimizer=optimizer, loss=['MeanSquaredError', 'categorical_crossentropy'],
                        loss_weights=[lambda_gen, lambda_pmf])


    # training
    combined_NN.fit(X_train,[Y_train_gen, Y_train_combined], epochs=1, batch_size=1, validation_data=(X_val, [Y_val_gen,Y_val_combined]), verbose=0)
    # fake image at step i ==> this is X in the paper and X_val is z in the paper
    fake_image = combined_NN(X_val)[0].numpy().reshape(32, 32, 3)

    # output probabilities at step i ==> this is J in the paper

    '''
    The warning: WARNING:tensorflow:Error in loading the saved optimizer state. As a result, your model is starting with a freshly initialized optimizer.
    This is happenning because this saved model optimizer state was not saved correctly which is not important in our case.
    '''


    trained_model = load_model("augmented_best_model.h5")
    output_vector_probabilities = trained_model(fake_image.reshape(1, 32,32,3)).numpy()[0]

    # D_2 distance between real image and fake image at step i==> this is equation (9)
    D_2_s = LA.norm(X_desired.reshape(3072,) - fake_image.reshape(3072,),          2)
    # SSIM distance between real image and fake image at step i ==>
    D_ssim_images = structural_similarity(X_desired, fake_image, multichannel = True)
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
        lambda_gen = relu_scaler_Ismail(lambda_gen       -   0.01 * 1     * ((D_ssim_images/delta_ssim)) * np.sign((D_ssim_images/delta_ssim)-1))
        lambda_pmf = relu_scaler_Ismail(lambda_pmf       -   0.001 * 0.001 * ((delta_js     /      D_JS)) * np.sign((delta_js     /D_JS      )-1))
    else:
        lambda_gen = 1
        lambda_pmf = 0.02



    ### SAVE THE DISTNCE AND PERTURBED IMAGE SO AS TO TAKE THE MINIMUM AT THE END OF THE TRAINING STEP (THIS IS TO OVER COME OVERFITTING DURING TRAINING)



fake_image = combined_NN(X_val)[0].numpy().reshape(32,32,3)
trained_model
### below is the same thing (just for sanity check)
trained_model = load_model("augmented_best_model.h5")
output_vector_probabilities   = trained_model(fake_image.reshape(1,32,32,3)).numpy()[0]

predicted_lbl_w_pert = np.argmax(output_vector_probabilities)

#output_vector_probabilities_2 = combined_NN(X_val)[1].numpy().reshape(10,)
# this is to make sure that above vectr are identical
#print('This vector MUST be zero',output_vector_probabilities-output_vector_probabilities_2)
#print(output_vector_probabilities_2)

real_image = X_desired.reshape(32,32,3)

plt.figure()
plt.subplot(2,2,1)
plt.title('Desired example')
plt.imshow(real_image,vmin=0, vmax=1)
plt.colorbar()
plt.axis('off')
plt.subplot(2,2,2)
plt.title('Generated example')
plt.imshow(fake_image,vmin=0, vmax=1)
plt.colorbar()
plt.axis('off')
plt.subplot(2,2,4)
plt.title('Generated example PMF')
plt.stem(output_vector_probabilities)
plt.ylim(top=1.2)
plt.subplot(2,2,3)
plt.title('Desired PMF')
plt.stem(Y_val_combined[0])
plt.ylim(top=1.2)


print('break')






