"""
2019-12-31 ResNet supported UNet
"""

from __future__ import division

from util import *

from keras import Input, models, layers, regularizers
from keras.optimizers import RMSprop,SGD, Adam
from keras import callbacks, losses
from keras import backend as K
from keras.utils import to_categorical
from keras.layers import  Conv2DTranspose, Lambda, Cropping1D, CuDNNGRU

from losses import *


# using the extanding and call the 2dTranspose function.  not work right the extra dimentional is kept
def Conv1DTranspose(input_tensor, filters, kernel_size, strides, padding='same'):
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x

####################################################################
#  Deep network basic blocks
####################################################################

def residual_block(x, kernel, filter_len, activation, stride, padding, BN=False):

    # BN
    conv = layers.BatchNormalization()(x) 
    conv = layers.Activation(activation)(conv)

    # two convolution
    conv = layers.Conv1D(kernel, filter_len, strides=stride, padding=padding, activation=activation)(conv)
    conv = layers.Conv1D(kernel, filter_len, strides=stride, padding=padding, activation=activation)(conv)

    conv = layers.Add()([conv, x])

    # additional BN
    if BN:
        conv = layers.BatchNormalization()(conv) 
        conv = layers.Activation(activation)(conv)

    return conv

##############################################################

def down_block_res(x, kernel, filter_len, activation, padding, initializer, stride, BN, maxpool_len):

    conv = layers.Conv1D(kernel, filter_len, strides=stride, padding=padding, kernel_initializer=initializer)(x)
    conv = residual_block(conv, kernel, filter_len, activation, stride, padding=padding)
    conv = residual_block(conv, kernel, filter_len, activation, stride, padding=padding, BN=True)
    pool = layers.MaxPooling1D(maxpool_len)(conv)

    return conv, pool

def bottom_block_res(x, kernel, filter_len, activation, padding, initializer, stride, BN, DropoutRate):

    conv = layers.Conv1D(kernel, filter_len, strides=stride, padding=padding, kernel_initializer=initializer)(x)
    conv = residual_block(conv, kernel, filter_len, activation, stride, padding=padding)
    conv = residual_block(conv, kernel, filter_len, activation, stride, padding=padding, BN=True)
    
    if DropoutRate > 0:
        conv = layers.Dropout(DropoutRate)(conv)

    return conv


def up_block_transpose_res(x, skip, kernel, filter_len, activation, padding, initializer, stride, BN, maxpool_len):

    up = Conv1DTranspose(x, kernel, filter_len, strides=maxpool_len, padding=padding)
    merge = layers.Concatenate(-1)([skip, up])

    # need extra conv to make the size consistent
    conv = layers.Conv1D(kernel, filter_len, strides=stride, padding=padding, kernel_initializer=initializer)(merge)

    conv = residual_block(conv, kernel, filter_len, activation, stride, padding=padding)
    conv = residual_block(conv, kernel, filter_len, activation, stride, padding=padding, BN=True)

    return conv

#################################################################
# backup usage#
def up_block_res(x, skip, kernel, filter_len, activation, padding, initializer, stride, BN, maxpool_len):

    up = layers.UpSampling1D(maxpool_len)(x)
    merge = layers.Concatenate(-1)([skip, up])

    conv = residual_block(merge, kernel, filter_len, activation, stride, padding=padding)
    conv = residual_block(conv, kernel, filter_len, activation, stride, padding=padding, BN=True)

    return conv

def up_block_transpose(x, skip, kernel, filter_len, activation, padding, initializer, stride, BN, maxpool_len):

    up = Conv1DTranspose(x, kernel, filter_len, strides=maxpool_len, padding=padding)
    merge = layers.Concatenate(-1)([skip, up])

    conv = layers.Conv1D(kernel, filter_len, strides=stride, padding=padding, kernel_initializer=initializer)(merge)
    if BN: conv = layers.BatchNormalization()(conv) 
    conv = layers.Activation(activation)(conv)
            
    conv = layers.Conv1D(kernel, filter_len, strides=stride, padding=padding, kernel_initializer=initializer)(conv)
    if BN: conv = layers.BatchNormalization()(conv)
    conv = layers.Activation(activation)(conv) 

    return conv
###################################################################

###################################################################
# Test and dev
###################################################################
## 4 layer 
def UNet_ResNet(rd_input, kernels, filter_len, maxpool_len, stride, BN=True, DropoutRate=0.2):
    
    initializer = 'he_normal' #'glorot_uniform'
    padding="same"
    stride = 1

    conv1, pool1 = down_block_res(rd_input, kernels[0], filter_len, "relu", padding, initializer, stride, BN, maxpool_len[0])
    conv2, pool2 = down_block_res(pool1,    kernels[1], filter_len, "relu", padding, initializer, stride, BN, maxpool_len[1])
    conv3, pool3 = down_block_res(pool2,    kernels[2], filter_len, "relu", padding, initializer, stride, BN, maxpool_len[2])
    conv4, pool4 = down_block_res(pool3,    kernels[3], filter_len, "relu", padding, initializer, stride, BN, maxpool_len[3])

    # Bottom, constains dropout
    conv_bottom = bottom_block_res(pool4, kernels[4], filter_len, "relu", padding, initializer, stride, BN, DropoutRate)

    # up-sampling
    up0 = up_block_transpose_res(conv_bottom, conv4, kernels[3], filter_len, "relu", padding, initializer, stride, BN, maxpool_len[3])
    up1 = up_block_transpose_res(up0,         conv3, kernels[2], filter_len, "relu", padding, initializer, stride, BN, maxpool_len[2])
    up2 = up_block_transpose_res(up1,         conv2, kernels[1], filter_len, "relu", padding, initializer, stride, BN, maxpool_len[1])
    up3 = up_block_transpose_res(up2,         conv1, kernels[0], filter_len, "relu", padding, initializer, stride, BN, maxpool_len[0])

    output = layers.Conv1D(1, 1, activation='sigmoid')(up3)

    model = models.Model(rd_input, output)
    return model


#Do ResNet block for the left part only.
def UNet_ResNet_left(rd_input, kernels, filter_len, maxpool_len, stride, BN=True, DropoutRate=0.2):
    
    initializer = 'he_normal' #'glorot_uniform'
    padding="same"
    stride = 1

    conv1, pool1 = down_block_res(rd_input, kernels[0], filter_len, "relu", padding, initializer, stride, BN, maxpool_len[0])
    conv2, pool2 = down_block_res(pool1,    kernels[1], filter_len, "relu", padding, initializer, stride, BN, maxpool_len[1])
    conv3, pool3 = down_block_res(pool2,    kernels[2], filter_len, "relu", padding, initializer, stride, BN, maxpool_len[2])
    conv4, pool4 = down_block_res(pool3,    kernels[3], filter_len, "relu", padding, initializer, stride, BN, maxpool_len[3])


    # Bottom, constains dropout
    conv_bottom = bottom_block_res(pool4, kernels[4], filter_len, "relu", padding, initializer, stride, BN, DropoutRate)

    # up-sampling
    up0 = up_block_transpose(conv_bottom, conv4, kernels[3], filter_len, "relu", padding, initializer, stride, BN, maxpool_len[3])
    up1 = up_block_transpose(up0,         conv3, kernels[2], filter_len, "relu", padding, initializer, stride, BN, maxpool_len[2])
    up2 = up_block_transpose(up1,         conv2, kernels[1], filter_len, "relu", padding, initializer, stride, BN, maxpool_len[1])
    up3 = up_block_transpose(up2,         conv1, kernels[0], filter_len, "relu", padding, initializer, stride, BN, maxpool_len[0])

    output = layers.Conv1D(1, 1, activation='sigmoid')(up3)

    model = models.Model(rd_input, output)
    return model


def ResNet(rd_input, kernels, filter_len, maxpool_len, stride, BN=True, DropoutRate=0.2):

    initializer = 'he_normal' #'glorot_uniform'
    padding="same"
    stride = 1

    conv1, pool1 = down_block_res(rd_input, kernels[0], filter_len, "relu", padding, initializer, stride, BN, maxpool_len[0])
    conv2, pool2 = down_block_res(pool1,    kernels[1], filter_len, "relu", padding, initializer, stride, BN, maxpool_len[1])
    conv3, pool3 = down_block_res(pool2,    kernels[2], filter_len, "relu", padding, initializer, stride, BN, maxpool_len[2])
    conv4, pool4 = down_block_res(pool3,    kernels[3], filter_len, "relu", padding, initializer, stride, BN, maxpool_len[3])


    # Bottom, constains dropout
    conv_bottom = bottom_block_res(pool4, kernels[4], filter_len, "relu", padding, initializer, stride, BN, DropoutRate)

    # fc layer
    output = layers.Flatten()(conv_bottom)
    output = layers.Dense(rd_input.shape[1].value, activation='sigmoid')(output)

    model = models.Model(rd_input, output)
    return model

