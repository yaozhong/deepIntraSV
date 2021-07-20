"""
Date: 2018-10-04
Author: Yao-zhong Zhang
Description: investigate the information gap amony different inputs.
1. original normalized read depth data
2. RD probability
3. GC-content normalization effect

# 2018-10-20: visualization Unet prediction output
"""

from __future__ import division
from util import *

from keras import Input, models, layers, regularizers
from keras.optimizers import RMSprop,SGD
from keras import callbacks
from keras import backend as K
from keras.utils import to_categorical
import shutil, os

from sklearn.decomposition import PCA


def visual_prediction(input_x, rgs, gold, pred, figurePathName):

    if os.path.exists(figurePathName):
        shutil.rmtree(figurePathName)
    
    os.mkdir(figurePathName)

    sample_num = pred.shape[0]
    
    if(len(input_x.shape) == 3):
        signal = (input_x[:,:,0]).reshape(input_x.shape[0], input_x.shape[1])
    

    if len(pred.shape) > 2:
        pred = pred.reshape(pred.shape[0], pred.shape[1])
    if len(gold.shape) > 2:
        gold = gold.reshape(gold.shape[0], gold.shape[1])


    for i in range(sample_num):

        # plot each figure in the fold

        if np.sum(gold[i]) == 0:
            filePath = figurePathName + "/bg-"+"_".join(rgs[i])+ ".png"
            figName = "Background bin,"
        else:
            filePath = figurePathName + "/breakPoint-"+"_".join(rgs[i]) + ".png"
            figName = "Breakpoint bin,"
        
        figName = figName + rgs[i][3] + ":"  + " Chr-"+rgs[i][0] + "(" + rgs[i][1] +"," + rgs[i][2] + ")" 

        fig = plt.figure()
        x = np.array(range(gold.shape[1]), dtype=np.int16) + 1

        plt.plot(x, signal[i,:] , color="black")
        plt.grid(True)

        #rectangle plot of gold and prediction
        cs = [ "red" if val==1 else "green"  for val in gold[i] ]
        plt.scatter(x, [-5]*len(gold[i]), color = cs, marker="s")

        cs = [ "red" if val==1 else "green" for val in pred[i] ]
        plt.scatter(x, [-7]*len(pred[i]), color = cs, marker="s")

        #print the position of break point of gold break point
        if rgs[i][3] != "BG":
            plt.axvline(x=int(rgs[i][5]), color='b', linestyle='--')

        plt.yticks((4,3,2,1,0,-1,-2,-3,-4, -5,-7), (4,3,2,1,0,-1,-2,-3,-4, "Gold", "Pred"))
        plt.title(figName)
        plt.savefig(filePath)
        plt.close("all")
    
    print "* The segment files are saved in ", figurePathName

