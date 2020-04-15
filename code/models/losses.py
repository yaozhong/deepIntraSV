# Copyright 2019 
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, You can obtain one at http://mozilla.org/MPL/2.0/.
#


from __future__ import division

from keras import callbacks, losses, metrics
from keras import backend as K
from keras.utils import to_categorical
import numpy as np
from tensorflow.python.ops import array_ops, math_ops
import tensorflow as tf

## Evluation metric for the final score calcuation
## 1. Editor distance
def edit_distance(y_true, y_pred):

    y_true = create_sparse(K.tf.argmax(y_true, axis=-1))
    y_pred = create_sparse(K.tf.argmax(y_pred, axis=-1))

    return(K.tf.edit_distance(y_pred, y_true, normalize=True))


def dice_score(gold, pred):
    gold, pred = gold.flatten(), pred.flatten()
    intersection = np.sum(gold*pred)
    df = (2. * intersection + np.finfo(np.float32).eps) / (np.sum(gold) + np.sum(pred) + np.finfo(np.float32).eps)
    return df

def iou_score(gold, pred):
    intersection = np.sum(gold*pred)
    df = (intersection + np.finfo(np.float32).eps) / (np.sum(gold) + np.sum(pred) - intersection + np.finfo(np.float32).eps)
    return df

# consider this one
#############################################
# segmentation loss
#############################################
def iou_coef(y_true, y_pred, smooth=K.epsilon()):

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    iou = (intersection + smooth) / ( union + smooth)
    return iou

def iou_loss(y_true, y_pred):
    return 1. - iou_coef(y_true, y_pred)



def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2.*intersection + K.epsilon()) / ( K.sum(y_true) + K.sum(y_pred) + K.epsilon())

def dice_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


##################################################
# mixture of different loss
##################################################

def ce_dice_loss(y_true, y_pred):
    return losses.categorical_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

## no weight is used
def bc_dice_loss(y_true, y_pred):
    return losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def abc_dice_loss(y_true, y_pred):
    return 0.5 * losses.binary_crossentropy(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred)

def abc_iou_loss(y_true, y_pred):
    return 0.5 * losses.binary_crossentropy(y_true, y_pred) + 0.5 * iou_loss(y_true, y_pred)

##################################################
# speical designed loss
##################################################

def bk_loss(y_true, y_pred):
    return weighted_cross_entropy(y_true, y_pred)

def weighted_cross_entropy(y_true, y_pred):
    try:
        [seg_unpack, weight_unpack] = K.tf.unstack(y_true, 2, axis=-1)      
        seg = K.tf.expand_dims(seg_unpack, -1)
        weight = K.tf.expand_dims(weight_unpack, -1)
    except:
        pass

    #
    #w_rmse = K.mean(math_ops.multiply(weight, math_ops.square(y_pred - seg)), axis=-1)

    # weighted _ binary cross-entropy
    epsilon = K.epsilon()
    y_pred = K.tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    zeros = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)
    cond = (y_pred >= zeros)
    relu_logits = math_ops.select(cond, y_pred, zeros)
    neg_abs_logits = math_ops.select(cond, -y_pred, y_pred)
    entropy = math_ops.add(relu_logits - y_pred * seg, math_ops.log1p(math_ops.exp(neg_abs_logits)), name=None)
    w_bc = K.mean(math_ops.multiply(weight, entropy), axis=-1) 

    # Dice loss
    seg_f = K.flatten(seg)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(seg_f * y_pred_f)
    dice_loss=  1. - (2.*intersection + K.epsilon()) / ( K.sum(seg_f) + K.sum(y_pred_f) + K.epsilon())
   
    return  dice_loss - w_bc


## refer binary logit binary loss function
"""
def weighted_cross_entropy(y_true, y_pred):
    try:
        [seg, weight] = K.tf.unstack(y_true, 2, axis=2)
        seg = K.tf.expand_dims(seg, -1)
        weight = K.tf.expand_dims(weight, -1)
    except:
        pass

    epsilon = K.tf.convert_to_tensor(10e-8, y_pred.dtype.base_dtype)
    y_pred = K.tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    y_pred = K.tf.log(y_pred / (1 - y_pred))

    zeros = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)
    cond = (y_pred >= zeros)
    relu_logits = math_ops.select(cond, y_pred, zeros)
    neg_abs_logits = math_ops.select(cond, -y_pred, y_pred)
    entropy = math_ops.add(relu_logits - y_pred * seg, math_ops.log1p(math_ops.exp(neg_abs_logits)), name=None)
    return K.mean(math_ops.multiply(weight, entropy), axis=-1) 
"""


### those are used for the basic evluation of unpaccking
def unpack_binary_accuracy(y_true, y_pred):
    try:
        [seg, weight] = K.tf.unstack(y_true, 2, axis=-1)
        
        seg = K.tf.expand_dims(seg, -1)
        weight = K.tf.expand_dims(weight, -1)
    except:
        pass

    score = metrics.binary_accuracy(seg, y_pred)
    return score

def unpack_dice_score(y_true, y_pred):
    try:
        [seg, weight] = K.tf.unstack(y_true, 2, axis=-1)
        
        seg = K.tf.expand_dims(seg, -1)
        #weight = K.tf.expand_dims(weight, -1)
    except:
        pass

    y_true_f = K.flatten(seg)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score=  (2.*intersection + K.epsilon()) / ( K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())
   
    return score

def unpack_iou_score(y_true, y_pred):
    try:
        [seg, weight] = K.tf.unstack(y_true, 2, axis=-1)
        
        seg = K.tf.expand_dims(seg, -1)
        weight = K.tf.expand_dims(weight, -1)
    except:
        pass

    score = iou_score(seg, y_pred)
    return score



## 20190607
def categorical_focal_loss(gamma=2., alpha=.25):
    """
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """
        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)
    return categorical_focal_loss_fixed


# 20180608 test version, wights, 
def weight_categorical_loss(weights):

    weights = K.variable(weights)
    
    def weight_categorical_loss_fix(y_true, y_pred):
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True) 
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return weight_categorical_loss_fix



