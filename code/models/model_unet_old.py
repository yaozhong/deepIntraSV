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
from keras_contrib.layers import CRF

import config

#################################################
## loss functions used for training Unet
#################################################

def dice_coef(y_true, y_pred):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f*y_pred_f)
    return (2.*intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

def dice_score(gold, pred):
    gold, pred = gold.flatten(), pred.flatten()
    intersection = np.sum(gold*pred)
    df = (2. * intersection + 1) / (np.sum(gold) + np.sum(pred) + 1)
    return df

def ce_dice_loss(y_true, y_pred):
    return losses.binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)



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



## old version, the parameters format is totally different.
def UNet_networkstructure_basic_old(rd_input, conv_window_len, maxpooling_len,BN=True, DropoutRate=0.2):
            
            initializer = 'he_normal' #'glorot_uniform'
            
            ##################### Conv1 #########################      
            conv1 = layers.Conv1D(64, conv_window_len, activation='relu', padding='same', \
                kernel_initializer=initializer)(rd_input)
            if BN: conv1 = layers.BatchNormalization()(conv1)
            #conv1 = layers.Activation('relu')(conv1)
            
            conv1 = layers.Conv1D(64, conv_window_len,  activation='relu', padding='same', \
                kernel_initializer=initializer)(conv1)
            if BN: conv1 = layers.BatchNormalization()(conv1) 
            
            #conv1 = layers.Activation('relu')(conv1)
            pool1 = layers.MaxPooling1D(maxpooling_len[0])(conv1)
        
            ##################### Conv2 ##########################
            conv2 = layers.Conv1D(128, conv_window_len, activation='relu', padding='same',\
                kernel_initializer=initializer)(pool1)
            if BN: conv2 = layers.BatchNormalization()(conv2) 
            #conv2 = layers.Activation('relu')(conv2)
            
            conv2 = layers.Conv1D(128, conv_window_len, activation='relu', padding='same',\
                kernel_initializer=initializer)(conv2)
            if BN: conv2 = layers.BatchNormalization()(conv2) 
            
            #conv2 = layers.Activation('relu')(conv2)
            pool2 = layers.MaxPooling1D(maxpooling_len[1])(conv2)

            ##################### conv3 ###########################
            conv3 = layers.Conv1D(256, conv_window_len, activation='relu', padding='same',\
                kernel_initializer=initializer)(pool2)
            if BN: conv3 = layers.BatchNormalization()(conv3) 
            #conv3 = layers.Activation('relu')(conv3)
            
            conv3 = layers.Conv1D(256, conv_window_len, activation='relu', padding='same',\
                kernel_initializer=initializer)(conv3)
            if BN: conv3 = layers.BatchNormalization()(conv3) 
            #conv3 = layers.Activation('relu')(conv3)
            if DropoutRate > 0:
                drop3 = layers.Dropout(DropoutRate)(conv3)
            else:
                drop3 = conv3
            pool3 = layers.MaxPooling1D(maxpooling_len[2])(drop3)


            ####################  conv4 (U bottle) #####################
            conv4 = layers.Conv1D(512, conv_window_len, activation='relu', padding='same',\
                kernel_initializer=initializer)(pool3)
            if BN: conv4 = layers.BatchNormalization()(conv4) 
            #conv4 = layers.Activation('relu')(conv4)
            
            conv4 = layers.Conv1D(512, conv_window_len, activation='relu', padding='same',\
                kernel_initializer=initializer)(conv4)
            if BN: conv4 = layers.BatchNormalization()(conv4) 
            #conv4 = layers.Activation('relu')(conv4)
            if DropoutRate > 0:
                drop4 = layers.Dropout(DropoutRate)(conv4)
            else:
                drop4 = conv4

            ################### upSampling, upConv5 ##########################
            up5 = layers.UpSampling1D(maxpooling_len[3])(drop4)
            merge5 = layers.Concatenate(-1)([drop3, up5])

            conv5 = layers.Conv1D(256, conv_window_len, activation='relu', padding='same', \
                kernel_initializer=initializer)(merge5)
            if BN: conv5 = layers.BatchNormalization()(conv5) 
            #conv5 = layers.Activation('relu')(conv5)
            
            conv5 = layers.Conv1D(256, conv_window_len, activation='relu', padding='same', \
                kernel_initializer=initializer)(conv5)
            if BN: conv5 = layers.BatchNormalization()(conv5) 


            ################### upConv 6 ##############################
            up6 = layers.UpSampling1D(maxpooling_len[4])(conv5)
            merge6 = layers.Concatenate(-1)([conv2, up6])
        
            conv6 = layers.Conv1D(128, conv_window_len, activation='relu', padding='same', \
                kernel_initializer=initializer)(merge6)
            if BN: conv6 = layers.BatchNormalization()(conv6) 
            #conv6 = layers.Activation('relu')(conv6)
            
            conv6 = layers.Conv1D(128, conv_window_len, activation='relu', padding='same',\
                kernel_initializer=initializer)(conv6)
            if BN: conv6 = layers.BatchNormalization()(conv6) 
            #conv6 = layers.Activation('relu')(conv6)


            ################### upConv 7 #########################
            up7 = layers.UpSampling1D(maxpooling_len[5])(conv6)
            merge7 = layers.Concatenate(-1)([conv1, up7])

            conv7 = layers.Conv1D(64, conv_window_len, activation= 'relu', padding='same',\
                kernel_initializer=initializer)(merge7)
            if BN: conv7 = layers.BatchNormalization()(conv7) 
            #conv7 = layers.Activation('relu')(conv7)
            
            conv7 = layers.Conv1D(64, conv_window_len, activation= 'relu', padding='same', \
                kernel_initializer=initializer)(conv7)
            if BN: conv7 = layers.BatchNormalization()(conv7) 
            #conv7 = layers.Activation('relu')(conv7)

            ################## final output ###################### 
            conv8 = layers.Conv1D(2, conv_window_len, activation= 'relu', padding='same', \
                kernel_initializer=initializer)(conv7)
            if BN: conv8 = layers.BatchNormalization()(conv8) 
            #conv8 = layers.Activation('relu')(conv8)
            
            if DropoutRate > 0:
                conv8 = layers.Dropout(DropoutRate)(conv8)
                
            conv9 = layers.Conv1D(1, 1, activation='sigmoid')(conv8)

            model = models.Model(rd_input, conv9)

            return model


# plus CRF version for the output
def UNet_networkstructure_crf(rd_input, conv_window_len, maxpooling_len,BN=True, DropoutRate=0.5):
            
            initializer = 'he_normal' #'glorot_uniform'
            
            ##################### Conv1 #########################      
            conv1 = layers.Conv1D(64, conv_window_len, activation='relu', padding='same', \
                kernel_initializer=initializer)(rd_input)
            if BN: conv1 = layers.BatchNormalization()(conv1)
            #conv1 = layers.Activation('relu')(conv1)
            
            conv1 = layers.Conv1D(64, conv_window_len,  activation='relu', padding='same', \
                kernel_initializer=initializer)(conv1)
            if BN: conv1 = layers.BatchNormalization()(conv1) 
            
            #conv1 = layers.Activation('relu')(conv1)
            pool1 = layers.MaxPooling1D(maxpooling_len[0])(conv1)
        
            ##################### Conv2 ##########################
            conv2 = layers.Conv1D(128, conv_window_len, activation='relu', padding='same',\
                kernel_initializer=initializer)(pool1)
            if BN: conv2 = layers.BatchNormalization()(conv2) 
            #conv2 = layers.Activation('relu')(conv2)
            
            conv2 = layers.Conv1D(128, conv_window_len, activation='relu', padding='same',\
                kernel_initializer=initializer)(conv2)
            if BN: conv2 = layers.BatchNormalization()(conv2) 
            
            #conv2 = layers.Activation('relu')(conv2)
            pool2 = layers.MaxPooling1D(maxpooling_len[1])(conv2)

            ##################### conv3 ###########################
            conv3 = layers.Conv1D(256, conv_window_len, activation='relu', padding='same',\
                kernel_initializer=initializer)(pool2)
            if BN: conv3 = layers.BatchNormalization()(conv3) 
            #conv3 = layers.Activation('relu')(conv3)
            
            conv3 = layers.Conv1D(256, conv_window_len, activation='relu', padding='same',\
                kernel_initializer=initializer)(conv3)
            if BN: conv3 = layers.BatchNormalization()(conv3) 
            #conv3 = layers.Activation('relu')(conv3)
            if DropoutRate > 0:
                drop3 = layers.Dropout(DropoutRate)(conv3)
            else:
                drop3 = conv3
            pool3 = layers.MaxPooling1D(maxpooling_len[2])(drop3)


            ####################  conv4 (U bottle) #####################
            conv4 = layers.Conv1D(512, conv_window_len, activation='relu', padding='same',\
                kernel_initializer=initializer)(pool3)
            if BN: conv4 = layers.BatchNormalization()(conv4) 
            #conv4 = layers.Activation('relu')(conv4)
            
            conv4 = layers.Conv1D(512, conv_window_len, activation='relu', padding='same',\
                kernel_initializer=initializer)(conv4)
            if BN: conv4 = layers.BatchNormalization()(conv4) 
            #conv4 = layers.Activation('relu')(conv4)
            if DropoutRate > 0:
                drop4 = layers.Dropout(DropoutRate)(conv4)
            else:
                drop4 = conv4

            ################### upSampling, upConv5 ##########################
            up5 = layers.UpSampling1D(maxpooling_len[3])(drop4)
            merge5 = layers.Concatenate(-1)([drop3, up5])

            conv5 = layers.Conv1D(256, conv_window_len, activation='relu', padding='same', \
                kernel_initializer=initializer)(merge5)
            if BN: conv5 = layers.BatchNormalization()(conv5) 
            #conv5 = layers.Activation('relu')(conv5)
            
            conv5 = layers.Conv1D(256, conv_window_len, activation='relu', padding='same', \
                kernel_initializer=initializer)(conv5)
            if BN: conv5 = layers.BatchNormalization()(conv5) 


            ################### upConv 6 ##############################
            up6 = layers.UpSampling1D(maxpooling_len[4])(conv5)
            merge6 = layers.Concatenate(-1)([conv2, up6])
        
            conv6 = layers.Conv1D(128, conv_window_len, activation='relu', padding='same', \
                kernel_initializer=initializer)(merge6)
            if BN: conv6 = layers.BatchNormalization()(conv6) 
            #conv6 = layers.Activation('relu')(conv6)
            
            conv6 = layers.Conv1D(128, conv_window_len, activation='relu', padding='same',\
                kernel_initializer=initializer)(conv6)
            if BN: conv6 = layers.BatchNormalization()(conv6) 
            #conv6 = layers.Activation('relu')(conv6)


            ################### upConv 7 #########################
            up7 = layers.UpSampling1D(maxpooling_len[5])(conv6)
            merge7 = layers.Concatenate(-1)([conv1, up7])

            conv7 = layers.Conv1D(64, conv_window_len, activation= 'relu', padding='same',\
                kernel_initializer=initializer)(merge7)
            if BN: conv7 = layers.BatchNormalization()(conv7) 
            #conv7 = layers.Activation('relu')(conv7)
            
            conv7 = layers.Conv1D(64, conv_window_len, activation= 'relu', padding='same', \
                kernel_initializer=initializer)(conv7)
            if BN: conv7 = layers.BatchNormalization()(conv7) 
            #conv7 = layers.Activation('relu')(conv7)

            ################## final output ###################### 
            conv8 = layers.Conv1D(2, conv_window_len, activation= 'relu', padding='same', \
                kernel_initializer=initializer)(conv7)
            if BN: conv8 = layers.BatchNormalization()(conv8) 
            #conv8 = layers.Activation('relu')(conv8)
            
            if DropoutRate > 0:
                conv8 = layers.Dropout(DropoutRate)(conv8)
            
            conv9 = layers.Conv1D(1, 1, activation='sigmoid')(conv8)
            crf = CRF(2, sparse_target=True)
            conv9 = crf(conv9)

            model = models.Model(rd_input, conv9)

            return (model, crf)


