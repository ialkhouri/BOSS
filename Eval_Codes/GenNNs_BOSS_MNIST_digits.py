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


#########################################################################################################################################
############################### SSIM fucntion  ######
#########################################################################################################################################
def SSIM_index(imageA, imageB):

    imageA = imageA.reshape(28, 28)
    imageB = imageB.reshape(28, 28)

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


################################################### below is the MNIST digits

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
# reshape data to fit model
X_train = train_images.reshape(train_images.shape[0], 28, 28, 1)
X_test = test_images.reshape(test_images.shape[0], 28, 28, 1)
X_train, X_test = X_train/255, X_test/255
# normalization:
train_images = train_images / 255
test_images = test_images / 255
print("")

y_train = np_utils.to_categorical(train_labels,10)
y_test = np_utils.to_categorical(test_labels,10)

X_test = X_test.astype(np.float32)



####################################################################################

# ################################ some dataset - MNIST fashion
#
# # download mnist data and split into train and test sets
# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
# # reshape data to fit model
# X_train = train_images.reshape(train_images.shape[0], 28, 28, 1)
# X_test = test_images.reshape(test_images.shape[0], 28, 28, 1)
# X_train, X_test = X_train/255, X_test/255
# # normalization:
# train_images = train_images / 255
# test_images = test_images / 255
# # ###############################################



## get a trained model (such as the MNIST didgits)

#trained_model = load_model("MNIST_digits__avgPool_dense_softmax_together_model.h5") # input is \in [0,1]
trained_model = load_model("MNIST_digits_trained_range_1to1_1d_input.h5") # input is \in [-1,1]; this model here has 1D input of f, hence we need generator NN to be of output 1,128


#trained_model = load_model("MNIST_digits_trained_model_3.h5") # input is \in [0,1]

#trained_model = load_model("MNIST_digits_trained_model_2.h5") # input is \in [-1,1]

# X_test  = 2*X_test  - 1
# results = trained_model.evaluate(X_test, y_test)
# print("test loss, test acc:", results)

#trained_model = load_model("MNIST_digits_trained_model_3.h5")
#trained_model = load_model("MNIST_digits_trained_model_gan_like.h5")

######## freeze trained_model
for layer in trained_model.layers:
    layer.trainable = False


########################################################################################
###########################################################################
#################################### BUILDING THE gen model g(z,\phi)
###########################################################################





gen_NN = tf.keras.Sequential()

## ADDING THE GEN MODEL layers that will be trained

layer = layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,), name='dense_gen')
layer.trainable=True
gen_NN.add(layer)

layer = layers.BatchNormalization()
layer.trainable=True
gen_NN.add(layer)

layer = layers.LeakyReLU()
layer.trainable=True
gen_NN.add(layer)

layer = layers.Reshape((7, 7, 256))
layer.trainable=True
gen_NN.add(layer)
#assert combined_NN.output_shape == (None, 7, 7, 256)

layer = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)
layer.trainable=True
gen_NN.add(layer)
#assert gen_NN.output_shape == (None, 14, 14, 64)

layer = layers.BatchNormalization()
layer.trainable=True
gen_NN.add(layer)

layer = layers.LeakyReLU()
layer.trainable=True
gen_NN.add(layer)

layer = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
layer.trainable=True
gen_NN.add(layer)
#assert gen_NN.output_shape == (None, 14, 14, 64)

layer = layers.BatchNormalization()
layer.trainable=True
gen_NN.add(layer)

layer = layers.LeakyReLU()
layer.trainable=True
gen_NN.add(layer)

layer = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
layer.trainable=True
gen_NN.add(layer)


# below is added for the 1D modification
layer = layers.Reshape((784, 1, 1))
layer.trainable=True
gen_NN.add(layer)




#########################################################################################################################################
############################### this is NOT sequentiail traning (traning two loss fucntions from different heads of the NN) ######
#########################################################################################################################################

### define X_d as the desired
# make the image in [-1,1]
X_desired = X_test[21]*2  -1


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
number_of_classes  = 10
target_class       =  test_labels[21]
desired_confidence =  0.5

#1 code the confidence only
desired_PMF_confidence = np.zeros(shape=(1,number_of_classes))
for i in range(number_of_classes):
    if i == target_class:
        desired_PMF_confidence[:,i] = desired_confidence
    else:
        desired_PMF_confidence[:,i] = (1-desired_confidence) / (number_of_classes-1)


#2 code the decision boundary examples between class i and class j
desired_PMF_boundary = np.zeros(shape=(1,number_of_classes))
class_i = 7
class_j = 6
#below is the values of PMF[i] and PMF[j] (i.e. maximum of 0.5)
desired_confidence_boundary = 0.5
for i in range(number_of_classes):
    if i == class_i or i == class_j:
        desired_PMF_boundary[:,i] = desired_confidence_boundary
    else:
        desired_PMF_boundary[:,i] = (1-(2*desired_confidence_boundary)) / (number_of_classes-2)


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

# # below is for the 2D image
# Y_train_np_gen = np.zeros(shape=(batch_size_gen,28,28,1))
# Y_val_gen = X_desired.reshape(1,28,28,1)
# for i in range(batch_size_gen):
#     Y_train_np_gen[i,:,:,:] = Y_val_gen.reshape(28,28,1)
# # convert Y_train to tf eager tensor
# Y_train_gen = tf.convert_to_tensor(Y_train_np_gen, dtype=tf.float32)


# below is for the 1D image
Y_train_np_gen = np.zeros(shape=(batch_size_gen,784,1,1))
Y_val_gen = X_desired.reshape(1,784,1,1)
for i in range(batch_size_gen):
    Y_train_np_gen[i,:,:,:] = Y_val_gen.reshape(784,1,1)
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

#Y_val_combined = np.array([[0.6,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0]])

Y_val_combined = desired_PMF_confidence

#Y_val_combined = np.array([[1,0,0,0,0,0,0,0,0,0]])

for i in range(batch_size_2):
    Y_train_combined[i,:] = Y_val_combined

Y_desired = Y_val_combined[0]



print('break')


####################################################################################################################
### defining the combined model such that its the concatenation of g, then f ==> this is defing model h in the paper
#############################################################################################################################

input = Input(shape=100)

### Calling the gen model and defining the first output (the out of model g)
#this is for the first gen


# x = gen_NN.layers[0](input)
# for lay in range(10):
#     layer = gen_NN.layers[lay+1]
#     layer.trainable = True
#     x = layer(x)
# out_1 = x


x = gen_NN.layers[0](input)
for lay in range(len(gen_NN.layers) - 1):
    layer = gen_NN.layers[lay+1]
    layer.trainable = True
    x = layer(x)
out_1 = x






# # this is for the second gen
# x = gen_NN.layers[0](input)
# for lay in range(5):
#     layer = gen_NN.layers[lay+1]
#     layer.trainable = True
#     x = layer(x)
# out_1 = x



#### Calling the trained model and defining the 2nd output (the output of f or h)
# x_2 = trained_model.layers[0](x)
# for lay in range(5):
#     layer = trained_model.layers[lay + 1]
#     layer.trainable = False
#     x_2 = layer(x_2)
# out_2 = x_2

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

### define the loss functions for each head
#combined_NN.compile(optimizer = optimizer, loss=['MeanSquaredError','categorical_crossentropy'],loss_weights=[5,0.15])

#combined_NN.compile(optimizer = optimizer, loss=['MeanAbsolutePercentageError','KLDivergence'],loss_weights=[5,0.15])

#combined_NN.compile(optimizer = optimizer, loss=['MeanSquaredError','MeanAbsolutePercentageError'],loss_weights=[1,1])

#combined_NN.compile(optimizer = optimizer, loss=['MeanAbsolutePercentageError','categorical_crossentropy'],loss_weights=[1.1,5])

loss_1 = tf.keras.losses.MeanSquaredError(name='LOSS_1')

loss_2 = tf.keras.losses.CategoricalCrossentropy(name='LOSS_2', from_logits=False,label_smoothing=0)

#loss_2 = tf.keras.losses.MeanSquaredLogarithmicError(name='LOSS_2')

#combined_NN.compile(optimizer = optimizer, loss=[loss_1,loss_2],loss_weights=[5,10])


#combined_NN.compile(optimizer = optimizer, loss=['MeanAbsolutePercentageError','MeanAbsoluteError'],loss_weights=[1,5])



# poisson and tanh at gen model does not work
#combined_NN.compile(optimizer = optimizer, loss=['Poisson','KLDivergence'],loss_weights=[5,0.11])

#combined_NN.compile(optimizer = optimizer, loss=['CosineSimilarity','KLDivergence'],loss_weights=[5,0.11])

# ### we need to enforce again in "combined_NN" to freeze weights for the trained model.
# for lay in range(18):
#     if lay >= 12:
#         layer = combined_NN.layers[lay]
#         layer.trainable = False


#combined_NN.fit(X_train,[Y_train_gen, Y_train_combined], epochs=10, batch_size=32, validation_data=(X_val, [Y_val_gen,Y_val_combined]), verbose=1)

dynamic_weights_selection = True
# initial losses functions weights
lambda_gen = 1
lambda_pmf = 0.02

############# trainING LOOP
for i in range(traning_steps):


    combined_NN.compile(optimizer=optimizer, loss=[loss_1, loss_2], loss_weights=[lambda_gen, lambda_pmf])
    # for lay in range(18):
    #     if lay >= 12:
    #         layer = combined_NN.layers[lay]
    #         layer.trainable = False


    # traning
    combined_NN.fit(X_train,[Y_train_gen, Y_train_combined], epochs=1, batch_size=1, validation_data=(X_val, [Y_val_gen,Y_val_combined]), verbose=0 )
    #combined_NN.train_on_batch(X_train, [Y_train_gen, Y_train_combined])
    # fake image at step i ==> this is X in the paper and X_val is z in the paper
    fake_image = combined_NN(X_val)[0].numpy().reshape(28, 28)


    # output probabilities at step i ==> this is J in the paper


    #trained_model = load_model("MNIST_digits_trained_model_3.h5")
    output_vector_probabilities = trained_model(fake_image.reshape(1, 28, 28, 1)).numpy()[0]

    #output_vector_probabilities = combined_NN(X_val)[1].numpy().reshape(10,)

    # D_2 distance between real image and fake image at step i==> this is equation (9)
    D_2_s = LA.norm(X_desired.reshape(784,) - fake_image.reshape(784,),          2)
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
        lambda_gen = relu_scaler_Ismail(lambda_gen       -   0.01 * 1    * ((D_ssim_images/delta_ssim)) * np.sign((D_ssim_images/delta_ssim)-1))
        lambda_pmf = relu_scaler_Ismail(lambda_pmf       -   0.05 * 0.02 * ((delta_js/D_JS))            * np.sign((delta_js/D_JS           )-1))
    else:
        lambda_gen = 1
        lambda_pmf = 0.02


    ### SAVE THE DISTNCE AND PERTURBED IMAGE SO AS TO TAKE THE MINIMUM AT THE END OF THE TRAINING STEP (THIS IS TO OVER COME OVERFITTING DURING TRAINING)



fake_image = combined_NN(X_val)[0].numpy().reshape(28,28)

### below is the same thing (just for sanity check)
trained_model = load_model("MNIST_digits_trained_range_1to1_1d_input.h5")
output_vector_probabilities   = trained_model(fake_image.reshape(1,28,28,1)).numpy()[0]
#output_vector_probabilities_2 = combined_NN(X_val)[1].numpy().reshape(10,)
# this is to make sure that above vectr are identical
#print('This vector MUST be zero',output_vector_probabilities-output_vector_probabilities_2)
#print(output_vector_probabilities_2)

real_image = X_desired.reshape(28,28)

plt.figure()
plt.subplot(2,2,1)
plt.title('Desired example')
plt.imshow(real_image,cmap='gray',vmin=-1, vmax=1)
plt.colorbar()
plt.axis('off')
plt.subplot(2,2,2)
plt.title('Generated example')
plt.imshow(fake_image,cmap='gray',vmin=-1, vmax=1)
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


#########################################################################################################################################
################## optimality condition KKT1 with similarity - Abu Ismail
#########################################################################################################################################

#### build the grad_x ( I(x,x_d) ):
N = 784
xx = fake_image.reshape(N)
yy = X_desired.reshape(N)

# calculate c_1, c_2
c1 = (0.01 * ( np.max(xx) - np.min(xx)) ) * (0.01 * ( np.max(xx) - np.min(xx)) )
c2 = (0.03 * ( np.max(xx) - np.min(xx)) ) * (0.03 * ( np.max(xx) - np.min(xx)) )

# calculate mu_x and mu_y
mu_x = np.mean(xx)
mu_y = np.mean(yy)

mu_xs = mu_x*mu_x
mu_ys = mu_y*mu_y

# calculate variance
var_x = np.var(xx)
var_y = np.var(yy)

# calculate covariance
covar = np.cov(xx,yy)[0,1]


T1 = ((2*mu_x*mu_y+c1)*(4*np.ones(N) - (4*np.ones(N)/N))) / ((mu_xs*mu_xs+c1)*(var_x+var_y+c2))

T2 = ((2*covar+c2)*(2*mu_x*mu_y+c1))*((2/N)*xx - (2*mu_x/N)*np.ones(N)) / ((var_x+var_y+c2)*(mu_xs*mu_xs*mu_ys*mu_ys + 2*c1*mu_xs*mu_ys+c1*c1))

T3 = (2*covar+c2)*((2/N)*np.ones(N)) / ((mu_xs*mu_ys+c1)*(var_x+var_y+c2))

T4 = (2*mu_x*mu_y+c1)*(2*covar+c2)*((4*mu_ys/N)*np.ones(N)) / ((var_x+var_y+c2)*((mu_xs*mu_xs*mu_ys*mu_ys + 2*c1*mu_xs*mu_ys+c1*c1)))

grad_sim = T1 -T2 + T3 - T4

# below is gget the gradients
grad_matrix     = np.zeros(shape=(784,10))
matrix_with_log = np.zeros(shape=(784,10))
temp = np.zeros(shape=(10,))
for i in range(10):
    temp[i] = np.log2(output_vector_probabilities[i]  /   (output_vector_probabilities[i]+Y_desired[i])  )
    #np.log2(output_vector_probabilities[i] / (output_vector_probabilities[i] + Y_desired[i]))
    grad_matrix[:,i]     = grad_discriminant_sm_wrt_1d_img(fake_image, i, trained_model).numpy()[0,:,:].reshape(784,)
    matrix_with_log[:,i] = temp[i] * grad_matrix[:,i]

sum_of_all_grads_case_1 = np.sum(matrix_with_log,1)

kk = np.zeros(shape=(784,))

mu_1_minus_mu_2 = 0.00025

kk = sum_of_all_grads_case_1 + mu_1_minus_mu_2 * grad_sim

opt_ball_2 = LA.norm(kk-np.zeros(shape=(784,)),                  2       )
opt_ball_i = LA.norm(kk-np.zeros(shape=(784,)),                  np.inf       )
print("KKT1: solution is of distance from zeros [L2,Linf] = ", [opt_ball_2,opt_ball_i])


# #########################################################################################################################################
# ################## optimality condition KKT1 with similarity
# #########################################################################################################################################
#
# #### build the grad_x ( I(x,x_d) ):
# N = 784
# xx = fake_image.reshape(N)
# yy = X_desired.reshape(N)
#
# # calculate c_1, c_2
# c1 = (0.01 * ( np.max(xx) - np.min(xx)) ) * (0.01 * ( np.max(xx) - np.min(xx)) )
# c2 = (0.03 * ( np.max(xx) - np.min(xx)) ) * (0.03 * ( np.max(xx) - np.min(xx)) )
#
# # calculate mu_x and mu_y
# mu_x = np.mean(xx)
# mu_y = np.mean(yy)
#
# # calculate variance
# var_x = np.var(xx)
# var_y = np.var(yy)
#
# # calculate covariance
# covar = np.cov(xx,yy)[0,1]
#
#
# term_11 = ( (2*mu_x*mu_y) + c1 )  /  ( (mu_x*mu_x)+(mu_y*mu_y)+c1 )
#
# term_12 = (   ((var_x+var_y+c2) * (4*(N-1)/N) * np.ones(N))   - (2*covar +c2)*(((2/N)*xx)) - (2*mu_x/N)*np.ones(N)   )  /  ( (var_y+var_x+c2)*(var_y+var_x+c2) )
#
# term_1 = term_11*term_12
#
# term_21 = ((2*covar)+c2)/((var_y+var_x+c2))
#
# term_22 = (    (( (mu_x*mu_x)+(mu_y*mu_y)+c1 ) *  (2/N)*mu_y)*np.ones(N)    -     (( (2*mu_x*mu_y) + c1 ) * (2*mu_x/N)*np.ones(N))   )  /      (( (mu_x*mu_x)+(mu_y*mu_y)+c1 )*( (mu_x*mu_x)+(mu_y*mu_y)+c1 ))
#
# term_2 = term_21*term_22
#
# grad_sim = term_1 + term_2
#
# # below is gget the gradients
# grad_matrix     = np.zeros(shape=(784,10))
# matrix_with_log = np.zeros(shape=(784,10))
# temp = np.zeros(shape=(10,))
# for i in range(10):
#     temp[i] = np.log2(output_vector_probabilities[i]  /   (output_vector_probabilities[i]+Y_desired[i])  )
#     #np.log2(output_vector_probabilities[i] / (output_vector_probabilities[i] + Y_desired[i]))
#     grad_matrix[:,i]     = grad_discriminant_sm_wrt_1d_img(fake_image, i, trained_model).numpy()[0,:,:].reshape(784,)
#     matrix_with_log[:,i] = temp[i] * grad_matrix[:,i]
#
# sum_of_all_grads_case_1 = np.sum(matrix_with_log,1)
#
# kk = np.zeros(shape=(784,))
#
# mu_1_minus_mu_2 = 0.00025
#
# kk = sum_of_all_grads_case_1 + mu_1_minus_mu_2 * grad_sim
#
# opt_ball_2 = LA.norm(kk-np.zeros(shape=(784,)),                  2       )
# opt_ball_i = LA.norm(kk-np.zeros(shape=(784,)),                  np.inf       )
# print("KKT1: solution is of distance from zeros [L2,Linf] = ", [opt_ball_2,opt_ball_i])


#########################################################################################################################################
################## optimality condition KKT1
#########################################################################################################################################
# case 0: define a \mu and make it case_ + \mu (x* - x_d)
mu = 0.025

# below is gget the gradients
grad_matrix     = np.zeros(shape=(784,10))
matrix_with_log = np.zeros(shape=(784,10))
temp = np.zeros(shape=(10,))
for i in range(10):
    temp[i] = np.log2(output_vector_probabilities[i]  /   (output_vector_probabilities[i]+Y_desired[i])  )
    #np.log2(output_vector_probabilities[i] / (output_vector_probabilities[i] + Y_desired[i]))
    grad_matrix[:,i]     = grad_discriminant_sm_wrt_1d_img(fake_image, i, trained_model).numpy()[0,:,:].reshape(784,)
    matrix_with_log[:,i] = temp[i] * grad_matrix[:,i]

sum_of_all_grads_case_1 = np.sum(matrix_with_log,1)
kk = np.zeros(shape=(784,))
kk = sum_of_all_grads_case_1 + 2*mu*( fake_image.reshape(784,) - X_desired.reshape(784,)  )


opt_ball_2 = LA.norm(kk-np.zeros(shape=(784,)),                  2       )
opt_ball_i = LA.norm(kk-np.zeros(shape=(784,)),                  np.inf       )
print("KKT1: solution is of distance from zeros [L2,Linf] = ", [opt_ball_2,opt_ball_i])

################## optimality condition KKT2
# we can say that KKT2 can be replaces by the SSIM index, hence report SSIM
#KKT2 = LA.norm(fake_image.reshape(784,) - X_desired.reshape(784,),2)*LA.norm(fake_image.reshape(784,) - X_desired.reshape(784,),2)



################## optimality condition KKT3
if mu >= 0:
    print("KKT2 is satisfied")









#
# opt_ball = LA.norm(sum_of_all_grads_case_1-np.zeros(shape=(784,)),                  2       )
# print("solution for case 1 is of L2 distance from zeros which is = ", opt_ball)
#
#
# sum_of_all_grads = np.sum(grad_matrix,1)
# # hence solution is of by the lp distance of sum_of_all_grads and vector of all zeros
# opt_ball = LA.norm(sum_of_all_grads-np.zeros(shape=(784,)),                  2       )
# print("solution for case 3 is of L2 distance from zeros which is = ", opt_ball)

# #########################################################################################################################################
# ################## optimality condition KKT1 case 3:
# #########################################################################################################################################
# # \sum_{i\in[M]} grad(J_i(X^*)) = 0
#
# # for x^* obtained from the above algorithm, get grad(J_i(X^*)) for i \in [M]
#
# # for below function to work, we need a model with 1-D input AND seperate last dense and activation
#
# grad_matrix = np.zeros(shape=(784,10))
# for i in range(10):
#     grad_matrix[:,i] = grad_discriminant_sm_wrt_1d_img(fake_image, i, trained_model).numpy()[0,:,:].reshape(784,)
#
# sum_of_all_grads = np.sum(grad_matrix,1)
# # hence solution is of by the lp distance of sum_of_all_grads and vector of all zeros
# opt_ball = LA.norm(sum_of_all_grads-np.zeros(shape=(784,)),                  2       )
# print("solution is of L2 distance from zeros which is = ", opt_ball)


print('break')



