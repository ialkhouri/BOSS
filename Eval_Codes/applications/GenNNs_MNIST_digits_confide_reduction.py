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
############################### gradient of the model f w.r.t input image function   ######
#########################################################################################################################################
def grad_discriminant_sm_wrt_1d_img(input_image, lbl, model):


    """
    :param input_image: this is a numpy array of size X_test whichi 28,28,1
    :param input_label: the index of the output dis function
    :param model: sequential keras model trained with 1D image
    :return: gradient - same size as the input image
    """

    extractor = tf.keras.Model(inputs=model.inputs,
                                outputs=[layer.output for layer in model.layers])

    # image processing: convert input image to tf variable
    input_image_2 = tf.Variable(input_image, name='input_image_var')
    # reshape input tf.variable to 4 dim
    input_image_3 = tf.reshape(input_image_2, [1, 784, 1])
    with tf.GradientTape(watch_accessed_variables=True) as tape:
        tape.watch(input_image_3)
        # get the actual outputs
        features = extractor(input_image_3)
        # output of last layer
        # dis_func = model1.predict(input_image.reshape(1,784,1))[0]
        # i have no clue what this does
        dis_func = features[-1]
        #         # i have no clue what this does
        #
        func_val = dis_func[0][lbl]

    grad = tape.gradient(func_val, input_image_3)
    return grad


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

X_test  = 2*X_test  - 1

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




####################################################################################
# ################################ some dataset - MNIST fashion

## get a trained model (such as the MNIST didgits)
trained_model = load_model("MNIST_digits_trained_range_1to1_1d_input.h5") # input is \in [-1,1]; this model here has 1D input of f, hence we need generator NN to be of output 1,128



# ###############################################################################################################################################################################
# ####################################################################################### results for clean enviroment ########################################################################################
# confidence_clean_MNIST = []
# JS_clean_MNIST = []
#
# for idx in range(200):
#     sample = X_test[idx,:,:,:]
#     prob = trained_model(sample.reshape(1,28,28,1))[0].numpy()
#     prediction_conf  = np.argmax(prob)
#     # if the sample were correctly classified
#     if prediction_conf == test_labels[idx]:
#         # get the confidence level:
#         confidence_clean_MNIST.append(np.max(prob))
#         # get the JS distance with the ont hot encoding of the true lbl
#         if np.max(prob) >= 0.9999:
#             prob = y_test[idx]
#
#         JD_dis = D_JS_PMFs(prob, y_test[idx])
#         JS_clean_MNIST.append(JD_dis)
#         # logger:
#         print('idx = ', idx, 'predicted lbl = ',prediction_conf,'tru lbl = ',test_labels[idx], 'confidence_clean_CIFAR = ',np.max(prob), 'JS = ',JD_dis)
#
# print('For clean env, we get [ Avg_confidence,Avg_JS] =  ', [np.mean(confidence_clean_MNIST),np.mean(JS_clean_MNIST)])




# ####################################################################################################################
# # ################################ saving predicetd confidence scores of the first 200 images w.r.t the trained model
# ####################################################################################################################
# import pickle
# confidence_1st_200_images = []
#
# clean_200_test_images = []
#
# for ii in range(200):
#     X_test_sample = X_test[ii] * 2 - 1
#     clean_200_test_images.append(X_test_sample)
#     #output_vector_probabilities = trained_model(X_test_sample.reshape(1,28,28,1)).numpy()[0]
#     #confidence_1st_200_images.append(np.max(output_vector_probabilities))
#



######## freeze trained_model
for layer in trained_model.layers:
    layer.trainable = False


############## here what to save:
perturbed_images_conf_red = []
prob_vectors_conf_red     = []
JS_conf_red = []
SSIM__conf_red = []

JS_conf_red_per_train_step=[]
SSIM_conf_per_train_step=[]
kk_2_per_train_step=[]
kk_i_per_train_step=[]
lambda_g_save = []
lambda_h_save = []

confidence_SCiTaP_C_MNIST = []
JS_SCiTaP_C_MNIST = []
SSIM_SCiTaP_C_MNIST = []



for idx in range(20):


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
    X_desired = X_test[idx]

    #X_desired = X_test[4]

    # these two need to be have the same values as of now since X_train is the same for both
    batch_size_gen = 80
    batch_size_2 = 80


    #################### training steps and stopping criteria
    delta_s  = 0.25


    delta_js = 0.15
    delta_ssim = 0.90


    delta_c = 0.25

    traning_steps = 20



    ############################################################
    ###  automated desired for confidence reduction y_d (desired PMF)
    ################################################################
    number_of_classes  = 10
    #target_class       =  6
    true_label = test_labels[idx]
    desired_confidence =  0.6

    #1 code the confidence only
    desired_PMF_confidence = np.zeros(shape=(1,number_of_classes))
    for i in range(number_of_classes):
        if i == true_label:
            desired_PMF_confidence[:,i] = desired_confidence
        else:
            desired_PMF_confidence[:,i] = (1-desired_confidence) / (number_of_classes-1)


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


    Y_train_combined = np.zeros(shape=(batch_size_2,10))

    Y_val_combined = desired_PMF_confidence

    for i in range(batch_size_2):
        Y_train_combined[i,:] = Y_val_combined

    Y_desired = Y_val_combined[0]






    ####################################################################################################################
    ### defining the combined model such that its the concatenation of g, then f ==> this is defing model h in the paper
    #############################################################################################################################

    input = Input(shape=100)

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

    ### define the loss functions for each head

    loss_1 = tf.keras.losses.MeanSquaredError(name='LOSS_1')

    loss_2 = tf.keras.losses.CategoricalCrossentropy(name='LOSS_2', from_logits=False,label_smoothing=0)


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
        trained_model = load_model("MNIST_digits_trained_range_1to1_1d_input.h5")
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
        #    print('BREAKING FOR IS USED with Distance SSIM = ', D_ssim_images, ' and D_JS = ', D_JS)
            break

        ### logger:
        print('training step = ', i, '; image SSIM = ', D_ssim_images, ' ; PMF_JS_Distance = ', D_JS, ' ; current loss weights = ', lambda_gen,' , ', lambda_pmf )

        ##### dynamic weight selection option in training
        if dynamic_weights_selection is True:
            lambda_gen = relu_scaler_(lambda_gen       -   0.01 * 1    * ((D_ssim_images/delta_ssim)) * np.sign((D_ssim_images/delta_ssim)-1))
            lambda_pmf = relu_scaler_(lambda_pmf       -   0.05 * 0.02 * ((delta_js/D_JS))            * np.sign((delta_js/D_JS           )-1))
        else:
            lambda_gen = 1
            lambda_pmf = 0.02

        #########################################################################################################################################
        ################## optimality condition KKT1
        #########################################################################################################################################
        # case 0: define a \mu and make it case_ + \mu (x* - x_d)
        mu = 0.025

        # below is gget the gradients
        grad_matrix = np.zeros(shape=(784, 10))
        matrix_with_log = np.zeros(shape=(784, 10))
        temp = np.zeros(shape=(10,))
        for ii in range(10):
            temp[ii] = np.log2(output_vector_probabilities[ii] / (output_vector_probabilities[ii] + Y_desired[ii]))
            # np.log2(output_vector_probabilities[i] / (output_vector_probabilities[i] + Y_desired[i]))
            grad_matrix[:, ii] = grad_discriminant_sm_wrt_1d_img(fake_image, ii, trained_model).numpy()[0, :, :].reshape(
                784, )
            matrix_with_log[:, ii] = temp[ii] * grad_matrix[:, ii]

        sum_of_all_grads_case_1 = np.sum(matrix_with_log, 1)
        kk = np.zeros(shape=(784,))
        kk = sum_of_all_grads_case_1 + 2 * mu * (fake_image.reshape(784, ) - X_desired.reshape(784, ))

        opt_ball_2 = LA.norm(kk - np.zeros(shape=(784,)), 2)
        opt_ball_i = LA.norm(kk - np.zeros(shape=(784,)), np.inf)

        ############## here what to save per training step:

        JS_conf_red_per_train_step.append([idx,i,D_JS])
        SSIM_conf_per_train_step.append([idx,i,D_ssim_images])
        kk_2_per_train_step.append([idx,i,opt_ball_2])
        kk_i_per_train_step.append([idx,i,opt_ball_i])
        lambda_g_save.append(lambda_gen)
        lambda_h_save.append(lambda_pmf)


        #print("KKT1: solution is of distance from zeros [L2,Linf] = ", [opt_ball_2, opt_ball_i])



#     fake_image = combined_NN(X_val)[0].numpy().reshape(28, 28)
#
#     # confidence_SCiTaP_C_MNIST = []
#     # JS_SCiTaP_C_MNIST = []
#     # SSIM_SCiTaP_C_MNIST = []
#
#
#     ### below is the same thing (just for sanity check)
#     trained_model = load_model("MNIST_digits_trained_range_1to1_1d_input.h5")
#     output_vector_probabilities = trained_model(fake_image.reshape(1, 28, 28, 1)).numpy()[0]
#
#     confidence = np.max(output_vector_probabilities)
#
#     confidence_SCiTaP_C_MNIST.append(confidence)
#     JS_SCiTaP_C_MNIST.append(D_JS)
#     SSIM_SCiTaP_C_MNIST.append(D_ssim_images)
#
#     print('idx = ', idx, 'is done with [conf,JS,ssim] = ', [confidence, D_JS, D_ssim_images])
#
#
#
# print(np.mean(confidence_SCiTaP_C_MNIST),np.mean(JS_SCiTaP_C_MNIST),np.mean(SSIM_SCiTaP_C_MNIST))



    print('Finished the image of index = ', idx,' and traing steps = ',i, ' with SSIM = ', D_ssim_images,
              ' and JS = ', [D_JS])

    # ############## here what to save:
    # perturbed_images_conf_red.append(fake_image)
    # prob_vectors_conf_red.append([output_vector_probabilities])
    # JS_conf_red.append([D_JS])
    # SSIM__conf_red.append([D_ssim_images])





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

# #########################################################################################################################################
# ################## optimality condition KKT1
# #########################################################################################################################################
# # case 0: define a \mu and make it case_ + \mu (x* - x_d)
# mu = 0.005
# # case 1
# # \sum_{i\in[M]} grad(J_i(X^*)) \log(J_i(X^*) / J_i(X^*)+y_i) = 0
#
# # case 3
# # \sum_{i\in[M]} grad(J_i(X^*)) = 0
#
# # below is gget the gradients
# grad_matrix     = np.zeros(shape=(784,10))
# matrix_with_log = np.zeros(shape=(784,10))
# temp = np.zeros(shape=(10,))
# for i in range(10):
#     temp[i] = np.log2(output_vector_probabilities[i]  /   (output_vector_probabilities[i]+Y_desired[i])  )
#     np.log2(output_vector_probabilities[i] / (output_vector_probabilities[i] + Y_desired[i]))
#     grad_matrix[:,i]     = grad_discriminant_sm_wrt_1d_img(fake_image, i, trained_model).numpy()[0,:,:].reshape(784,)
#     matrix_with_log[:,i] = temp[i] * grad_matrix[:,i]
#
# sum_of_all_grads_case_1 = np.sum(matrix_with_log,1)
# sum_of_all_grads_case_0 = np.zeros(shape=(784,))
# sum_of_all_grads_case_0 = sum_of_all_grads_case_1 + mu*( fake_image.reshape(784,) - X_desired.reshape(784,)  )
#
#
# opt_ball = LA.norm(sum_of_all_grads_case_0-np.zeros(shape=(784,)),                  2       )
# print("solution for case 0 is of L2 distance from zeros which is = ", opt_ball)
#
#
#
# opt_ball = LA.norm(sum_of_all_grads_case_1-np.zeros(shape=(784,)),                  2       )
# print("solution for case 1 is of L2 distance from zeros which is = ", opt_ball)
#
#
# sum_of_all_grads = np.sum(grad_matrix,1)
# # hence solution is of by the lp distance of sum_of_all_grads and vector of all zeros
# opt_ball = LA.norm(sum_of_all_grads-np.zeros(shape=(784,)),                  2       )
# print("solution for case 3 is of L2 distance from zeros which is = ", opt_ball)
#
# # #########################################################################################################################################
# # ################## optimality condition KKT1 case 3:
# # #########################################################################################################################################
# # # \sum_{i\in[M]} grad(J_i(X^*)) = 0
# #
# # # for x^* obtained from the above algorithm, get grad(J_i(X^*)) for i \in [M]
# #
# # # for below function to work, we need a model with 1-D input AND seperate last dense and activation
# #
# # grad_matrix = np.zeros(shape=(784,10))
# # for i in range(10):
# #     grad_matrix[:,i] = grad_discriminant_sm_wrt_1d_img(fake_image, i, trained_model).numpy()[0,:,:].reshape(784,)
# #
# # sum_of_all_grads = np.sum(grad_matrix,1)
# # # hence solution is of by the lp distance of sum_of_all_grads and vector of all zeros
# # opt_ball = LA.norm(sum_of_all_grads-np.zeros(shape=(784,)),                  2       )
# # print("solution is of L2 distance from zeros which is = ", opt_ball)
#
#
print('break')
#
#
#
