"""
2018-11-1 model structures of different implementation of U-net
"""

from __future__ import division

from util import *
from keras import Input, models, layers, regularizers
from keras.optimizers import RMSprop,SGD, Adam
from keras import callbacks, losses
from keras import backend as K
from keras.utils import to_categorical
from keras.layers import  Conv2DTranspose, Lambda, Cropping1D, CuDNNGRU
from keras_contrib.layers import CRF

import config
from losses import *


def getCropShape(target, refer):
    cw = (refer.get_shape()[1] -target.get_shape()[1]).value
    assert (cw >= 0)

    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)

    return (cw1, cw2)


def getCropShape_adj(target, refer, adj):

    cw = -(target.get_shape()[1] - refer.get_shape()[1]).value + adj
    print cw

    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1 
    else:
        cw1, cw2 = int(cw/2), int(cw/2)

    return (cw1, cw2)

# using the extanding and call the 2dTranspose function.  not work right the extra dimentional is kept
def Conv1DTranspose(input_tensor, filters, kernel_size, strides, padding='same'):
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x

####################################################################
#  Deep network basic blocks
####################################################################
def  UNet(rd_input, kernels, conv_window_len, maxpooling_len, stride, BN=True, DropoutRate=0.2):
            
            initializer = 'he_normal' #'glorot_uniform'
            padding_method="same"
            
            ##################### Conv1 #########################      
            conv1 = layers.Conv1D(kernels[0], conv_window_len, activation='relu', padding=padding_method, \
                kernel_initializer=initializer)(rd_input)
            if BN: conv1 = layers.BatchNormalization()(conv1)
            
            conv1 = layers.Conv1D(kernels[0], conv_window_len,  activation='relu', padding=padding_method, \
                kernel_initializer=initializer)(conv1)
            if BN: conv1 = layers.BatchNormalization()(conv1) 
            
            pool1 = layers.MaxPooling1D(maxpooling_len[0])(conv1)
        
            ##################### Conv2 ##########################
            conv2 = layers.Conv1D(kernels[1], conv_window_len, activation='relu', padding=padding_method,\
                kernel_initializer=initializer)(pool1)
            if BN: conv2 = layers.BatchNormalization()(conv2) 
       
            conv2 = layers.Conv1D(kernels[1], conv_window_len, activation='relu', padding=padding_method,\
                kernel_initializer=initializer)(conv2)
            if BN: conv2 = layers.BatchNormalization()(conv2) 
            
            pool2 = layers.MaxPooling1D(maxpooling_len[1])(conv2)

            ##################### conv3 ###########################
            conv3 = layers.Conv1D(kernels[2], conv_window_len, activation='relu', padding=padding_method,\
                kernel_initializer=initializer)(pool2)
            if BN: conv3 = layers.BatchNormalization()(conv3) 
     
            conv3 = layers.Conv1D(kernels[2], conv_window_len, activation='relu', padding=padding_method,\
                kernel_initializer=initializer)(conv3)
            if BN: conv3 = layers.BatchNormalization()(conv3) 

            if DropoutRate > 0:
                drop3 = layers.Dropout(DropoutRate)(conv3)
            else:
                drop3 = conv3

            pool3 = layers.MaxPooling1D(maxpooling_len[2])(drop3)


            ####################  conv4 (U bottle) #####################
            conv4 = layers.Conv1D(kernels[3], conv_window_len, activation='relu', padding=padding_method,\
                kernel_initializer=initializer)(pool3)
            if BN: conv4 = layers.BatchNormalization()(conv4) 
      
            conv4 = layers.Conv1D(kernels[3], conv_window_len, activation='relu', padding=padding_method,\
                kernel_initializer=initializer)(conv4)
            if BN: conv4 = layers.BatchNormalization()(conv4) 
            if DropoutRate > 0:
                drop4 = layers.Dropout(DropoutRate)(conv4)
            else:
                drop4 = conv4

            ################### upSampling, upConv5 ##########################
            up5 = layers.UpSampling1D(maxpooling_len[2])(drop4)
            merge5 = layers.Concatenate(-1)([drop3, up5])

            conv5 = layers.Conv1D(kernels[2], conv_window_len, activation='relu', padding=padding_method, \
                kernel_initializer=initializer)(merge5)
            if BN: conv5 = layers.BatchNormalization()(conv5) 
            
            conv5 = layers.Conv1D(kernels[2], conv_window_len, activation='relu', padding=padding_method, \
                kernel_initializer=initializer)(conv5)
            if BN: conv5 = layers.BatchNormalization()(conv5) 

            ################### upConv 6 ##############################
            up6 = layers.UpSampling1D(maxpooling_len[1])(conv5)
            merge6 = layers.Concatenate(-1)([conv2, up6])
        
            conv6 = layers.Conv1D(kernels[1], conv_window_len, activation='relu', padding=padding_method, \
                kernel_initializer=initializer)(merge6)
            if BN: conv6 = layers.BatchNormalization()(conv6) 
            
            conv6 = layers.Conv1D(kernels[1], conv_window_len, activation='relu', padding=padding_method,\
                kernel_initializer=initializer)(conv6)
            if BN: conv6 = layers.BatchNormalization()(conv6) 

            ################### upConv 7 #########################
            up7 = layers.UpSampling1D(maxpooling_len[0])(conv6)
            merge7 = layers.Concatenate(-1)([conv1, up7])

            conv7 = layers.Conv1D(kernels[0], conv_window_len, activation= 'relu', padding=padding_method,\
                kernel_initializer=initializer)(merge7)
            if BN: conv7 = layers.BatchNormalization()(conv7) 
            
            conv7 = layers.Conv1D(kernels[0], conv_window_len, activation= 'relu', padding=padding_method, \
                kernel_initializer=initializer)(conv7)
            if BN: conv7 = layers.BatchNormalization()(conv7) 

            ################## final output ###################### 
            # this part can be removed
            conv8 = layers.Conv1D(2, conv_window_len, activation= 'relu', padding=padding_method, \
                kernel_initializer=initializer)(conv7)
            if BN: conv8 = layers.BatchNormalization()(conv8) 
            
            if DropoutRate > 0:
                conv8 = layers.Dropout(DropoutRate)(conv8)
                
            conv9 = layers.Conv1D(1, 1, activation='sigmoid')(conv8)

            model = models.Model(rd_input, conv9)

            return model

