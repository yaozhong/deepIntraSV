"""
Date: 2018-08-16
Author: Yao-zhong Zhang
Description: the basic training for the keras model for the break point detection problem.

2018-09-19: revised the model for the multi-class classification and evluation metrics.
2018-10-09: Add the autoencoder for the classification task. [with/without attention]
"""

from __future__ import division

from keras import Input, models, layers, regularizers
from keras.optimizers import RMSprop,SGD, Adam
from keras import callbacks, losses
from keras import backend as K
from keras.utils import to_categorical

from keras_contrib.layers import CRF

from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn import svm
from sklearn.manifold import TSNE

from visualization import *
from mpl_toolkits.mplot3d import Axes3D
from model_unet import *
from model_hyperOpt_CNN import *

from dataProcess.data_cacheLoading import *


TRAIL=100
EPOCH=30

# model training settings

CB = [ callbacks.EarlyStopping(monitor="val_loss", patience=10, mode = "auto", restore_best_weights=True) ] 
VB = 0

CB1 = [ callbacks.EarlyStopping(monitor="val_acc", patience=30)]
CB2 = [ callbacks.EarlyStopping(monitor="val_viterbi_acc", patience=50, restore_best_weights=True) ] 

USESEQ = False
USEPROB = False
GC_NORM = False

def binary_eval(gold_cnv, pred_cnv, modelInfo, binaryCase=False):

        if binaryCase == False:
            gold_label = np.apply_along_axis(checkLabel, 1, gold_cnv)
            pred_label = np.apply_along_axis(checkLabel, 1, pred_cnv)

        else:
            gold_label = gold_cnv
            pred_label = pred_cnv

        # based on the gold standard split
        index1 = [ idx for idx in range(gold_label.shape[0]) if gold_label[idx] == 1]
        index0 = [ idx for idx in range(gold_label.shape[0]) if gold_label[idx] == 0]

        print("Test BK=%d / Total=%d" %(len(index1), len(gold_label)))
        print "[" + modelInfo + "]"

        # dice score section
        if binaryCase == False:
            print "\n---------- [Segmentation Results] -----------"
            df_cnv = dice_score(gold_cnv, pred_cnv)
            print(">> All dice_coef=%.4f" %(df_cnv))
            df1 = dice_score(gold_cnv[index1], pred_cnv[index1])   
            print("* BK only dice_coef=%.4f" %(df1))
            if len(index0) > 0:
                df0 = dice_score(gold_cnv[index0], pred_cnv[index0])           
                print("* Background dice_coef=%.4f" %(df0))
            print "-"* 30


        print "\n---------- [Binary Results] -----------"
        fscore = f1_score(gold_label, pred_label, average="micro")
        print ("* F-score=%f" %(fscore))
        tn, fp, fn, tp = confusion_matrix(gold_label, pred_label).ravel()       

        if len(index0) > 0 :
            fpr, tpr, thresholds = roc_curve(gold_label, pred_label)
            auc_value = auc(fpr, tpr)
            print("* AUC=%f" %auc_value)
            print("* Sensitivity=%f" %(tp/(tp+fn)))
            print("* FDR=%f" %(fp/(fp+tp)))

        print "-"* 30
        print confusion_matrix(gold_label, pred_label) 
        print "="*30
        if binaryCase == True:
            return (fscore, auc_value, tp/(tp+fn), fp/(fp+tp))
        else:
            return (df_cnv, df1, df0, fscore, auc_value, tp/(tp+fn), fp/(fp+tp))

"""
1-vs-1 SVM model
"""

def SVM_CV(dataPath, bk_dataPath, modelParamPath, modelSavePath, dataInfo, plotResult= False, plotTrainCurve=True):

        # model parameters

        modelInfo = "SVM-CV"

        x_data, y_data, rgs_data, x_cnv, y_cnv, rgs_cnv = loadData(dataPath, bk_dataPath, prob_add=USEPROB, seq_add=USESEQ)
        
        y_data = y_data.reshape(y_data.shape[0], y_data.shape[1], 1)
        y_data_label = np.apply_along_axis(checkLabel, 1, y_data)
        y_data_label = y_data_label.reshape(y_data_label.shape[0])
        
        x_data = x_data.reshape(x_data.shape[0], x_data.shape[1])
        y_data = y_data_label

        ## K-fold cross validation
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.DATABASE["rand_seed"])
        cv_scores, cv_bk_scores, cv_bg_scores, cv_auc, cnv_scores= [], [], [], [], []
        cv_sensitivity, cv_FDR = [], []

        index = 0
        for train_idx, test_idx in kfold.split(x_data, y_data_label): 

            index = index + 1

            model = svm.SVC()       # 1-vs-1
            model.fit(x_data[train_idx], y_data[train_idx])


            ###############################################################
            #### CV-split test set
            #################################################################
            t = model.predict(x_data[test_idx]).ravel()
            pred = t
            gold = y_data[test_idx].astype(np.float32)

            # save the prediction figures
            if plotResult == True:
                figureSavePath= "../experiment/result/"+ modelInfo + "_DATA-"+dataInfo+"-"+str(index)
                visual_prediction(x_data[test_idx], rgs_data[test_idx], gold, pred, figureSavePath)

            fscore, auc_value, sensitivity, FDR = binary_eval(gold, pred, modelInfo, True)

            cv_auc.append(auc_value)
            cv_sensitivity.append(sensitivity)
            cv_FDR.append(FDR)

        ####################################  Generating Results Report #################################
        
        print "\n=========== [DATA/MODEL information] ============="
        print "[-CV-" + dataInfo+ "]"
        print "["+ modelSavePath + "]"
        print "["+ modelInfo + "]"
            
        print '-'*30
        print("-- CV AUC %.4f (%.4f)" % (np.mean(cv_auc), np.std(cv_auc)))
        print("-- CV Sensitivity %.4f (%.4f)" % (np.mean(cv_sensitivity), np.std(cv_sensitivity)))
        print("-- CV FDR %.4f (%.4f)" % (np.mean(cv_FDR), np.std(cv_FDR)))
        print '*'*30


def RuleBased(dataPath, bk_dataPath, modelParamPath, dataInfo, modelInfo):

        x_data, y_data, rgs_data, x_cnv, y_cnv, rgs_cnv = loadData(dataPath, bk_dataPath, prob_add=USEPROB, seq_add=USESEQ)
        
        y_data_label = np.apply_along_axis(checkLabel, 1, y_data)
        bg_idx = [ idx for idx in range(len(y_data_label)) if y_data_label[idx] == 0]
        bk_idx = [ idx for idx in range(len(y_data_label)) if y_data_label[idx] == 1]
        
        print np.mean(x_data[bg_idx]), np.std(x_data[bg_idx])
        print np.mean(x_data[bk_idx]), np.std(x_data[bg_idx])


        x_data_shift = np.absolute(x_data[:, 1:x_data.shape[1]] - x_data[:,0:(x_data.shape[1]-1)])
        x_cnv_shift =  np.absolute(x_cnv[:,1:x_cnv.shape[1]] - x_cnv[:,0:(x_cnv.shape[1]-1)])

        x_data_diffs = np.array([ float(np.max(x_data_shift[i])) for i in range(x_data.shape[0])])
        x_cnv_diffs = np.array([ float(np.max(x_cnv_shift[i])) for i in range(x_cnv.shape[0])])

        print np.min(x_data_diffs[bg_idx]), np.max(x_data_diffs[bg_idx]), np.mean(x_data_diffs[bg_idx])
        print np.min(x_data_diffs[bk_idx]), np.max(x_data_diffs[bk_idx]), np.mean(x_data_diffs[bk_idx])


        # plot hist distribution
        fig = plt.figure()
        plt.hist(x_data_diffs[bg_idx], color="blue", density=True, bins=np.arange(np.min(x_data_diffs[bg_idx]), np.max(x_data_diffs[bg_idx]), 0.1))
        plt.hist(x_data_diffs[bk_idx], color="red", density=True, bins=np.arange(np.min(x_data_diffs[bk_idx]), np.max(x_data_diffs[bk_idx]),0.1))
        plt.savefig("../experiment/difference_hist.png")
        plt.close("all")




def SVM(dataPath, bk_dataPath, modelParamPath, dataInfo, modelInfo):


        x_data, y_data, rgs_data, x_cnv, y_cnv, rgs_cnv = loadData(dataPath, bk_dataPath, prob_add=USEPROB, seq_add=USESEQ)


        y_data = y_data.reshape(y_data.shape[0], y_data.shape[1], 1)
        y_cnv = y_cnv.reshape(y_cnv.shape[0], y_cnv.shape[1], 1)
        y_data_label = np.apply_along_axis(checkLabel, 1, y_data)
        y_cnv_label = np.apply_along_axis(checkLabel, 1, y_cnv)

        x_train = x_data.reshape(x_data.shape[0], x_data.shape[1])
        x_test = x_cnv.reshape(x_cnv.shape[0], x_cnv.shape[1])

        y_data_label = y_data_label.reshape(y_data_label.shape[0])
        y_cnv_label = y_cnv_label.reshape(y_cnv_label.shape[0])
        
        y_train = y_data_label 
        y_test = y_cnv_label
    
        """
        sidx = int(x_train.shape[0]*0.1)
        x_train = x_train[sidx:]
        y_train = y_train[sidx:]
        """

        model = svm.SVC()       # 1-vs-1
        model.fit(x_train, y_train)
            
        t = model.predict(x_test).ravel()


        print "\n=========== [DATA/MODEL information] ============="
        print "[" + dataInfo+ "]"
        print("Train BK=%d / Total=%d" %(np.sum(y_data_label), y_data_label.shape[0]))
        
        binary_eval(y_test, t, modelInfo, True)


# higher segment, but worse in classification
def CNN_networkstructure_v1(kernel_size, window_len, maxpooling_len, BN=True, DropoutRate=0.2):

        model = models.Sequential()
        # 1000bp (32,10,10) -> ()
        model.add(layers.Conv1D(kernel_size[0], window_len[0], padding="valid", activation="relu"))
        if BN: model.add(layers.BatchNormalization())
        
        model.add(layers.Conv1D(kernel_size[0], window_len[0], padding="valid", activation="relu"))
        if BN: model.add(layers.BatchNormalization())
        
        model.add(layers.MaxPooling1D(maxpooling_len[0]))


        model.add(layers.Conv1D(kernel_size[1], window_len[1], padding="valid", activation="relu"))
        if BN: model.add(layers.BatchNormalization())
        
        model.add(layers.Conv1D(kernel_size[1], window_len[1], padding="valid", activation="relu"))
        if BN: model.add(layers.BatchNormalization())
        
        model.add(layers.MaxPooling1D(maxpooling_len[1]))

        model.add(layers.Flatten())
        model.add(layers.Dropout(DropoutRate))
        model.add(layers.Dense(config.DATABASE["binSize"], activation='sigmoid'))
    

# classical 
def CNN_networkstructure(kernel_size, window_len, maxpooling_len, BN=True, DropoutRate=0.2):

        model = models.Sequential()
        # 1000bp (32,10,10) -> ()
        model.add(layers.Conv1D(kernel_size[0], window_len[0], activation="relu"))
        #if BN: model.add(layers.BatchNormalization())
        

        model.add(layers.Conv1D(kernel_size[1], window_len[1],activation="relu"))
        #model.add(layers.Conv1D(kernel_size[0], window_len[0], strides=window_len[0],activation="relu"))
        #if BN: model.add(layers.BatchNormalization())
        
        model.add(layers.MaxPooling1D(maxpooling_len[0]))


        #model.add(layers.Conv1D(kernel_size[1], window_len[1], strides=window_len[1], padding="valid",activation="relu"))
        #if BN: model.add(layers.BatchNormalization())
        #model.add(layers.Conv1D(kernel_size[1], window_len[1], strides=window_len[1], padding="valid",activation="relu"))
        #if BN: model.add(layers.BatchNormalization())
        
        #model.add(layers.MaxPooling1D(maxpooling_len[1]))

        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dropout(DropoutRate))
        model.add(layers.Dense(config.DATABASE["binSize"], activation='sigmoid'))
    
        return model


# 2019-01-04 add the CNN version for evluation
def CNN(dataPath, bk_dataPath, modelParamPath, modelSavePath, dataInfo, plotTrainCurve=True, plotResult=False):

        # Data loading start
        x_data, y_data, rgs_data, x_cnv, y_cnv, rgs_cnv = loadData(dataPath, bk_dataPath, prob_add=USEPROB, seq_add=USESEQ, gc_norm=GC_NORM)

        # for the convolutional output
        y_data = y_data.reshape(y_data.shape[0], y_data.shape[1], 1)
        y_data_label = np.apply_along_axis(checkLabel, 1, y_data)
        print("BK=%d / Total=%d" %(np.sum(y_data_label), y_data_label.shape[0]))       

        # model parameters loading
        if config.DATABASE["model_param"] != "":
            params = load_modelParam(config.DATABASE["model_param"])
        else:
            model_param_file  = modelParamPath + ".TRAIL-" + str(TRAIL) + ".CNN.model.parameters.json"
        
            # check whether the model is hyperOpt tuned
            if not os.path.exists(model_param_file):
                tmpData = (x_data, y_data, y_data_label)
                do_hyperOpt_CNN(tmpData, TRAIL, model_param_file)
        
            params = load_modelParam(model_param_file)

        print "*"*40
        print params
        print "*"*40

        # model parameters
        window_len = params["window_len"]              #[10, 5]
        kernel_size = params["kernel_size"]             # [64, 128]
        maxpooling_len = params["maxpooling_len"]
        lr = params["lr"]
        batchSize = params["batchSize"]
        dropoutRate = params["DropoutRate"]
        BN = params["BN"]

        modelInfo = "CNN_window_Len_"+"-".join([str(l) for l in window_len]) 
        modelInfo += "-kernel_size_"+ "-".join([str(l) for l in kernel_size])
        modelInfo += "-lr_"+ str(lr)
        modelInfo += "-batchSize" + str(batchSize)
        modelInfo += "-epoch"+str(params["epoch"])
        modelInfo += "-dropout"+str(dropoutRate)
        modelInfo += "-BN-"+str(BN)

        if USEPROB: modelInfo += "_USEPROB"
        if USESEQ: modelInfo += "_USESEQ"
        if GC_NORM: modelInfo += "_GC-NORM"

        #########################################
        modelSaveName = os.path.basename(modelParamPath) + "|" +modelInfo

        rd_input = Input(shape=(x_data.shape[1], x_data.shape[-1]), dtype='float32', name="rd")
        model = CNN_networkstructure(kernel_size, window_len, maxpooling_len, BN, dropoutRate)
        model.compile(optimizer=Adam(lr = lr) , loss= dice_coef_loss, metrics=[dice_coef])
        history = model.fit(x_data, y_data, epochs=params["epoch"], batch_size=batchSize, verbose=VB \
        ,callbacks=CB, validation_split=0.2)
        
        print "@ Saving model ..."
        model.save(modelSavePath +"/"+ modelSaveName +".h5" )
        model.summary()


        ###########################################################################################3
        # Results generation
        ###########################################################################################
        # plot the training curve:
        if plotTrainCurve:
            figureSavePath= modelSavePath + modelInfo + "_trainCurve.png"
            plt.plot(history.history['dice_coef'])
            plt.plot(history.history['val_dice_coef'])
            plt.title('Training curve of dice_coefficient')
            plt.xlabel('epoch')
            plt.ylabel('dice_coef')
            plt.legend(['train', 'valid'], loc='upper left')
            plt.savefig(figureSavePath)
            plt.close("all")

        """
        figureSavePath= "../figures/trainCurve/current/"+ dataSetName +"-bin_" + str(config.DATABASE["binSize"])+"-Unet_LossCurve_all" 
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Training curve for the loss')
        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.legend(['train', 'valid'], loc='upper left')
        plt.savefig(figureSavePath)
        plt.close("all")
        """

        ############# dice evluation #################
        t_cnv = model.predict(x_cnv, verbose=VB)
        pred_cnv = (t_cnv > 0.5).astype(np.float32).reshape(t_cnv.shape[0], t_cnv.shape[1])
        gold_cnv = y_cnv.astype(np.float32)

        # save the prediction figures
        if plotResult == True:
            figureSavePath= "../experiment/result/"+ modelInfo + "_DATA-"+dataInfo

            visual_prediction(x_cnv, rgs_cnv, gold_cnv, pred_cnv, figureSavePath)


        print "\n=========== [DATA/MODEL information] ============="
        print "[" + dataInfo+ "]"
        print("Train BK=%d / Total=%d" %(np.sum(y_data_label), y_data_label.shape[0]))

        binary_eval(gold_cnv, pred_cnv, modelInfo)


def CNN_CV(dataPath, bk_dataPath, modelParamPath, modelSavePath, dataInfo, plotResult= False, plotTrainCurve=False):

        # all data loading
        x_data, y_data, rgs_data, x_cnv, y_cnv, rgs_cnv = loadData(dataPath, bk_dataPath, prob_add=USEPROB, seq_add=USESEQ, gc_norm=GC_NORM)
        
        y_data = y_data.reshape(y_data.shape[0], y_data.shape[1], 1)
        y_data_label = np.apply_along_axis(checkLabel, 1, y_data)
        print("BK=%d / Total=%d" %(np.sum(y_data_label), y_data_label.shape[0]))
 
        # model parameters loading
        if config.DATABASE["model_param"] != "":
            params = load_modelParam(config.DATABASE["model_param"])
        else:

            model_param_file  =  modelParamPath + ".UNet.model.parameters.json"
            # check whether the model is hyperOpt tuned
            if not os.path.exists(model_param_file):
                tmpData = (x_data, y_data, y_data_label)
                do_hyperOpt(tmpData, TRAIL, model_param_file)

            params = load_modelParam(model_param_file)
        
        print "*"*40
        print params
        print "*"*40

        # model parameters
        window_len = params["window_len"]              #[10, 5]
        kernel_size = params["kernel_size"]             # [64, 128]
        maxpooling_len = params["maxpooling_len"]
        lr = params["lr"]
        batchSize = params["batchSize"]
        dropoutRate = params["DropoutRate"]
        BN = params["BN"]
        
        modelInfo = "CNN_window_Len_"+"-".join([str(l) for l in window_len]) 
        modelInfo += "-kernel_size_"+ "-".join([str(l) for l in kernel_size])
        modelInfo += "-lr_"+ str(lr)
        modelInfo += "-batchSize" + str(batchSize)
        modelInfo += "-epoch"+str(params["epoch"])
        modelInfo += "-dropout"+str(dropoutRate)
        modelInfo += "-BN-"+str(BN)

        if USEPROB: modelInfo += "_USEPROB"
        if USESEQ: modelInfo += "_USESEQ"


        ## K-fold cross validation
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.DATABASE["rand_seed"])
        cv_scores, cv_bk_scores, cv_bg_scores, cv_auc, cnv_scores= [], [], [], [], []
        cv_sensitivity, cv_FDR = [], []

        index = 0
        for train_idx, test_idx in kfold.split(x_data, y_data_label): 

            index = index + 1

            rd_input = Input(shape=(x_data.shape[1], x_data.shape[-1]), dtype='float32', name="rd")
            
            model = CNN_networkstructure(kernel_size, window_len, maxpooling_len, BN, dropoutRate)
            model.compile(optimizer=Adam(lr = lr) , loss= dice_coef_loss, metrics=[dice_coef])
            
            history = model.fit(x_data[train_idx], y_data[train_idx], epochs=params["epoch"], batch_size= batchSize, verbose=0, \
                    validation_split=0.2, callbacks=CB)
            
            ###########################################################################################3
            # Results generation
            ###########################################################################################
            if plotTrainCurve:

                figureSavePath= modelSavePath + modelInfo + "_trainCurve-"+str(index)+".png"
                plt.plot(history.history['dice_coef'])
                plt.plot(history.history['val_dice_coef'])
                plt.title('Training curve of dice_coefficient')
                plt.xlabel('epoch')
                plt.ylabel('dice_coef')
                plt.legend(['train', 'valid'], loc='upper left')
                plt.savefig(figureSavePath)
                plt.close("all")

            # evaluate the model
            ###############################################################
            #### CV-split test set
            #################################################################
            t = model.predict(x_data[test_idx], verbose=VB)
            pred = (t > 0.5).astype(np.float32)
            gold = y_data[test_idx].astype(np.float32)

            # save the prediction figures
            if plotResult == True:
                figureSavePath= "../experiment/result/"+ modelInfo + "_DATA-"+dataInfo+"-"+str(index)
                visual_prediction(x_data[test_idx], rgs_data[test_idx], gold, pred, figureSavePath)

            df, df1, df0, fscore, auc_value, sensitivity, FDR = binary_eval(gold, pred, modelInfo, False)

            cv_scores.append(df)
            cv_bk_scores.append(df1)
            cv_bg_scores.append(df0)
            
            cv_auc.append(auc_value)
            cv_sensitivity.append(sensitivity)
            cv_FDR.append(FDR)

        ####################################  Generating Results Report #################################
        
        print "\n=========== [DATA/MODEL information] ============="
        print "[-CV-" + dataInfo+ "]"
        print "["+ modelSavePath + "]"
        print "["+ modelInfo + "]"

        print '-'*30
        print("* CV all score %.4f (%.4f)" % (np.mean(cv_scores), np.std(cv_scores)))
        print("-- CV BK score %.4f (%.4f)" % (np.mean(cv_bk_scores), np.std(cv_bk_scores)))
        print("-- CV BG score %.4f (%.4f)" % (np.mean(cv_bg_scores), np.std(cv_bg_scores)))
            
        print '-'*30
        print("-- CV AUC %.4f (%.4f)" % (np.mean(cv_auc), np.std(cv_auc)))
        print("-- CV Sensitivity %.4f (%.4f)" % (np.mean(cv_sensitivity), np.std(cv_sensitivity)))
        print("-- CV FDR %.4f (%.4f)" % (np.mean(cv_FDR), np.std(cv_FDR)))
        print '*'*30



# non-cross validation setting for the u-net, first tailing

def UNet(dataPath, bk_dataPath, modelParamPath, modelSavePath, dataInfo, plotTrainCurve=True, plotResult=False):

        # Data loading start
        x_data, y_data, rgs_data, x_cnv, y_cnv, rgs_cnv = loadData(dataPath, bk_dataPath, prob_add=USEPROB, seq_add=USESEQ, gc_norm=GC_NORM)


## basic version
def BiLSTM(dataPath, bk_dataPath, modelSavePath, dataInfo, plotTrainCurve=True, plotResult=True):

        # model parameters
        lr = 1e-4
        batchSize = 8

        modelInfo = "BiLSTM-lr_"+ str(lr)
        modelInfo += "-batchSize" + str(batchSize)
        modelInfo += "-epoch"+str(EPOCH)

        if USEPROB: modelInfo += "_USEPROB"
        if USESEQ: modelInfo += "_USESEQ"

        # loading data part
        x_data, y_data, rgs_data, x_cnv, y_cnv, rgs_cnv = loadData(dataPath, bk_dataPath, prob_add=USEPROB, seq_add=USESEQ)
        

        # y_data_1hot = to_categorical(y_data)

        # for the convolutional output
        y_data = y_data.reshape(y_data.shape[0], y_data.shape[1], 1)
        y_data_label = np.apply_along_axis(checkLabel, 1, y_data)
        print("BK=%d / Total=%d" %(np.sum(y_data_label), y_data_label.shape[0]))       

        model = models.Sequential()
        model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
        model.add(layers.TimeDistributed(layers.Dense(1, activation='sigmoid')))

        model.compile(optimizer=Adam(lr = lr) , loss= dice_coef_loss, metrics=[dice_coef])
        history = model.fit(x_data, y_data, epochs=EPOCH, batch_size=batchSize, verbose=VB, \
                  validation_split=0.1, callbacks=CB)

        # plot the training curve:
        if plotTrainCurve:
            figureSavePath= modelSavePath + modelInfo + "_trainCurve.png"
            fig=plt.figure()
            plt.plot(history.history['dice_coef'])
            plt.plot(history.history['val_dice_coef'])
            plt.title('Training curve of dice_coefficient')
            plt.xlabel('epoch')
            plt.ylabel('dice_coef')
            plt.legend(['train', 'valid'], loc='upper left')
            plt.savefig(figureSavePath)
            plt.close("all")

        ############# dice evluation #################
        t_cnv = model.predict(x_cnv, verbose=VB)
        pred_cnv = (t_cnv > 0.5).astype(np.float32).reshape(t_cnv.shape[0], t_cnv.shape[1])
        gold_cnv = y_cnv.astype(np.float32)

        # save the prediction figures
        if plotResult == True:
            figureSavePath= "../experiment/result/"+ modelInfo + "_DATA-"+dataInfo

            visual_prediction(x_cnv, rgs_cnv, gold_cnv, pred_cnv, figureSavePath)
        

        print "\n=========== [DATA/MODEL information] ============="
        print "[" + dataInfo+ "]"
        print("Train BK=%d / Total=%d" %(np.sum(y_data_label), y_data_label.shape[0]))

        binary_eval(gold_cnv, pred_cnv, modelInfo)
        




def BiLSTM_crf(dataPath, bk_dataPath, modelSavePath, dataInfo, plotTrainCurve=True, plotResult=True):

        # model parameters
        lr = 1e-4
        batchSize = 8

        modelInfo = "BiLSTM_CRF-lr_"+ str(lr)
        modelInfo += "-batchSize" + str(batchSize)
        modelInfo += "-epoch"+str(EPOCH)

        if USEPROB: modelInfo += "_USEPROB"
        if USESEQ: modelInfo += "_USESEQ"

        # loading data part
        x_data, y_data, rgs_data, x_cnv, y_cnv, rgs_cnv = loadData(dataPath, bk_dataPath, prob_add=USEPROB, seq_add=USESEQ)
        
        # y_data_1hot = to_categorical(y_data)

        # for the convolutional output
        y_data = y_data.reshape(y_data.shape[0], y_data.shape[1], 1)
        y_data_label = np.apply_along_axis(checkLabel, 1, y_data)
        print("BK=%d / Total=%d" %(np.sum(y_data_label), y_data_label.shape[0]))       

        model = models.Sequential()
        model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
        model.add(layers.TimeDistributed(layers.Dense(1, activation='sigmoid')))
        crf = CRF(2, sparse_target=True)
        model.add(crf)

        model.compile(optimizer=Adam(lr = lr) , loss= crf.loss_function, metrics=[crf.accuracy])
        #model.compile(optimizer=Adam(lr = lr) , loss= dice_coef_loss, metrics=[dice_coef])
        #model.compile(optimizer='adam', loss=crf.loss_function, metric=[crf.accuracy])
        history = model.fit(x_data, y_data, epochs=EPOCH, batch_size=batchSize, verbose=VB, \
                  validation_split=0.1, callbacks=CB2)


        # plot the training curve:
        if plotTrainCurve:
            figureSavePath= modelSavePath + modelInfo + "_trainCurve.png"
            plt.plot(history.history['viterbi_acc'])
            plt.plot(history.history['val_viterbi_acc'])
            plt.title('Training curve of viterbi_acc')
            plt.xlabel('epoch')
            plt.ylabel('viterbi_acc')
            plt.legend(['train', 'valid'], loc='upper left')
            plt.savefig(figureSavePath)


        ############# dice evluation #################
        #t_cnv = model.predict(x_cnv, verbose=VB)
        pred_cnv = model.predict(x_cnv,verbose=VB).argmax(-1)
        #pred_cnv = (t_cnv > 0.5).astype(np.float32).reshape(t_cnv.shape[0], t_cnv.shape[1])
        gold_cnv = y_cnv.astype(np.float32)

        # save the prediction figures
        if plotResult == True:
            figureSavePath= "../experiment/result/"+ modelInfo + "_DATA-"+dataInfo

            visual_prediction(x_cnv, rgs_cnv, gold_cnv, pred_cnv, figureSavePath)



        print "\n=========== [DATA/MODEL information] ============="
        print "[" + dataInfo+ "]"
        print("Train BK=%d / Total=%d" %(np.sum(y_data_label), y_data_label.shape[0]))

        binary_eval(gold_cnv, pred_cnv, modelInfo)



# need to revised: directly loading basic U-net structures, not do hyperOpt, pre-fixed!
def UNet_crf(dataPath, bk_dataPath, modelParamPath, modelSavePath, dataInfo, plotTrainCurve=True, plotResult=True):

        # model parameters
        model_param_file  =  modelParamPath + ".UNet.model.parameters.json"
        
        if not os.path.exists(model_param_file):
            print "[Error!!]Not find UNet Parameter file! Please generate in single run model first!"
            return(-1)

        params = load_modelParam(model_param_file)
        print params   

        maxpooling_len = params["maxpooling_len"]
        conv_window_len = params["conv_window_len"]
        lr = params["lr"]
        batchSize = params["batchSize"]
        dropoutRate = params["DropoutRate"]

        modelInfo = "UNet_crf-all_maxpoolingLen_"+"-".join([str(l) for l in maxpooling_len]) 
        modelInfo += "-convWindowLen_"+ str(conv_window_len)
        modelInfo += "-lr_"+ str(lr)
        modelInfo += "-batchSize" + str(batchSize)
        modelInfo += "-epoch"+str(params["epoch"])
        modelInfo += "-dropout"+str(dropoutRate)

        if USEPROB: modelInfo += "_USEPROB"
        if USESEQ: modelInfo += "_USESEQ"


        # Data loading start
        x_data, y_data, rgs_data, x_cnv, y_cnv, rgs_cnv = loadData(dataPath, bk_dataPath, prob_add=USEPROB, seq_add=USESEQ)

        # for the convolutional output
        y_data = y_data.reshape(y_data.shape[0], y_data.shape[1], 1)
        y_data_label = np.apply_along_axis(checkLabel, 1, y_data)
        print("BK=%d / Total=%d" %(np.sum(y_data_label), y_data_label.shape[0]))       

        rd_input = Input(shape=(x_data.shape[1], x_data.shape[-1]), dtype='float32', name="rd")
        model,crf = UNet_networkstructure_crf(rd_input, conv_window_len, maxpooling_len, True, 0.2)
        model.compile(optimizer=Adam(lr = lr) , loss= crf.loss_function, metrics=[crf.accuracy])
        history = model.fit(x_data, y_data, epochs=EPOCH, batch_size=batchSize, verbose=VB, \
                  validation_split=0.1, callbacks=CB2)

        model.summary()
            
        ###########################################################################################3
        # Results generation
        ###########################################################################################
        # plot the training curve:
        if plotTrainCurve:
            figureSavePath= modelSavePath + modelInfo + "_trainCurve.png"
            plt.plot(history.history['viterbi_acc'])
            plt.plot(history.history['val_viterbi_acc'])
            plt.title('Training curve of viterbi_acc')
            plt.xlabel('epoch')
            plt.ylabel('viterbi_acc')
            plt.legend(['train', 'valid'], loc='upper left')
            plt.savefig(figureSavePath)
            plt.close("all")
        
        ############# Note the decoding SV
        #t_cnv = model.predict(x_cnv, verbose=VB)
        pred_cnv = model.predict(x_cnv, verbose=VB).argmax(-1)
        #pred_cnv = (t_cnv > 0.5).astype(np.float32).reshape(t_cnv.shape[0], t_cnv.shape[1])
        
        gold_cnv = y_cnv.astype(np.float32)

        # save the prediction figures
        if plotResult == True:
            figureSavePath= "../experiment/result/"+ modelInfo + "_DATA-"+dataInfo
            visual_prediction(x_cnv, rgs_cnv, gold_cnv, pred_cnv, figureSavePath)


        
        print "\n=========== [DATA/MODEL information] ============="
        print "[" + dataInfo+ "]"
        print("Train BK=%d / Total=%d" %(np.sum(y_data_label), y_data_label.shape[0]))

        binary_eval(gold_cnv, pred_cnv, modelInfo)





######################################################################
# The following functions is not tested!!
#####################################################################

# 2018-10-13 implementation, consider the related paper for the full-end model.
def AE(dataPath, bk_dataPath, modelName, niter=20):

        x_data, y_data = loadData(dataPath, bk_dataPath, shiftValue=USESHIFT, prob_add=False, seq_add=False)
        x_data = x_data.reshape(x_data.shape[0], x_data.shape[1])

        ## K-fold cross validation
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.DATABASE["rand_seed"])
        cvscores = []

        for train_idx, test_idx in kfold.split(x_data, y_data):

            # y_data_1hot = to_categorical(y_data)

            #1. add a model to train an AE
            ## 2. get the feature representation for each input and training another classifier ()
            rd_input = Input(shape=(x_data.shape[1],), dtype='float32', name="rd")
            encoder = layers.Dense(128, activation='relu')(rd_input)
            #encoder = layers.Dense(32, activation='relu')(encoder)
            encoder_output = layers.Dense(DIM_ENCODE)(encoder)

            decoder = layers.Dense(128, activation='relu')(encoder_output)
            #decoder = layers.Dense(64, activation='relu')(decoder)
            decoder = layers.Dense(x_data.shape[1], activation='tanh')(decoder)

            ae = models.Model(rd_input, decoder)
            encode = models.Model(rd_input, encoder_output)

            ae.compile(optimizer='adam', loss='mse', metrics=['mse'])
            ae.fit(x_data[train_idx], x_data[train_idx], epochs=EPOCH, batch_size=128, verbose=VB, validation_split=0.1, callbacks=CB)
            

            # get the middle layer vector
            x_data_encode = encode.predict(x_data[train_idx])

            ### construct classifier using
        
            svm_model = svm.SVC()       # 1-vs-1
            #model = svm.LinearSVC() # 1-vs-rest
            svm_model.fit(x_data_encode, y_data[train_idx])
            
            t = svm_model.predict(encode.predict(x_data[test_idx])).ravel()
            #print t
            acc=accuracy_score(y_data[test_idx], t)
            fscore = f1_score(y_data[test_idx], t, average="micro")
            print("* f1=%.4f, acc=%.2f" %(fscore,acc*100))
            cvscores.append(fscore)
            
            print confusion_matrix(y_data[test_idx], t) 

        print("%.4f (+/- %.4f)" % (np.mean(cvscores), np.std(cvscores)))

        #testEval_multi(model, x_test, y_test_origin, rgs_test)




def convRNN(dataPath, bk_dataPath, modelName, niter=20):


        x_data, y_data, rgs_data, x_cnv, y_cnv, rgs_cnv = loadData(dataPath, bk_dataPath, shiftValue=USESHIFT, prob_add=USEPROB, seq_add=USESEQ)

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.DATABASE["rand_seed"])
    
        y_data = y_data.reshape(y_data.shape[0], y_data.shape[1], 1)
        y_data_label = np.apply_along_axis(checkLabel, 1, y_data)
        print("BK=%d / Total=%d" %(np.sum(y_data_label), y_data_label.shape[0]))

        x_train = x_data.reshape(x_data.shape[0], x_data.shape[1], 1)
        y_train = y_data_label 

        cvscores = []
    
        for train_idx, test_idx in kfold.split(x_train, y_train):
            
            y_train_1hot = to_categorical(y_train)

            model = models.Sequential()
    
            model.add(layers.Conv1D(64, 3, strides=1, activation="relu"))
            model.add(layers.Conv1D(128, 3, strides=1, activation="relu"))
            model.add(layers.MaxPooling1D(2))

            model.add(layers.Bidirectional(layers.LSTM(64)))
            model.add(layers.Dense(N_class, activation='softmax'))

            model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(x_train[train_idx], y_train_1hot[train_idx], epochs=EPOCH, batch_size=128, verbose=VB, validation_split=0.1, callbacks=CB)

            # evaluate the model
            t = model.predict(x_train[test_idx])
            t = np.argmax(t, axis=1)   
            acc=accuracy_score(y_train[test_idx], t)
            fscore = f1_score(y_train[test_idx], t, average="micro")
            print("*f1=%.4f, acc=%.2f" %(fscore,acc*100))

            fpr, tpr, thresholds = roc_curve(y_train[test_idx], t)
            auc_value = auc(fpr, tpr)
            print ("-- AUC score is %f", auc_value)
            cvscores.append(auc_value)
        
            print confusion_matrix(y_train[test_idx], t) 


        print("%.4f (+/- %.4f)" % (np.mean(cvscores), np.std(cvscores)))




def LM(dataPath, bk_dataPath, modelName, niter=20):

        x_train, y_train = loadData(dataPath, bk_dataPath, shiftValue=USESHIFT, prob_add=False, seq_add=USESEQ)
    
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1])

        ## K-fold cross validation
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.DATABASE["rand_seed"])
        cvscores = []
    
        for train_idx, test_idx in kfold.split(x_train, y_train):
            
            y_train_1hot = to_categorical(y_train)

            model = models.Sequential()
    
            #model.add(layers.Dense(128))
            #model.add(layers.Dropout(0.5))
            model.add(layers.Dense(N_class, activation='softmax'))

            model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(x_train[train_idx], y_train_1hot[train_idx], epochs=EPOCH, batch_size=128, verbose=VB, validation_split=0.1, callbacks=CB)

            # evaluate the model
            t = model.predict(x_train[test_idx])
            t = np.argmax(t, axis=1)   
            acc=accuracy_score(y_train[test_idx], t)
            fscore = f1_score(y_train[test_idx], t, average="micro")
            print("*f1=%.4f, acc=%.2f" %(fscore,acc*100))
            cvscores.append(fscore)
            
            print confusion_matrix(y_train[test_idx], t) 

        print("%.4f (+/- %.4f)" % (np.mean(cvscores), np.std(cvscores)))



def MLP(dataPath, bk_dataPath, modelName, niter=20):

        x_train, y_train = loadData(dataPath, bk_dataPath, shiftValue=USESHIFT, prob_add=False, seq_add=USESEQ)
    
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1])

        ## K-fold cross validation
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.DATABASE["rand_seed"])
        cvscores = []
    
        for train_idx, test_idx in kfold.split(x_train, y_train):
            
            y_train_1hot = to_categorical(y_train)

            model = models.Sequential()
            
            #model.add(layers.Dense(128))
            #model.add(layers.Activation('relu'))
            #model.add(layers.Dropout(0.2))
            
            model.add(layers.Dense(64, activation='relu'))
            #model.add(layers.Dropout(0.2))

            model.add(layers.Dense(N_class, activation='softmax'))

            model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(x_train[train_idx], y_train_1hot[train_idx], epochs=EPOCH, batch_size=128, verbose=VB, validation_split=0.1, callbacks=CB)

            # evaluate the model
            t = model.predict(x_train[test_idx])
            t = np.argmax(t, axis=1)   
            acc=accuracy_score(y_train[test_idx], t)
            fscore = f1_score(y_train[test_idx], t, average="micro")
            print("*f1=%.4f, acc=%.2f" %(fscore,acc*100))
            cvscores.append(fscore)
        
            print confusion_matrix(y_train[test_idx], t) 

        print("%.4f (+/- %.4f)" % (np.mean(cvscores), np.std(cvscores)))



def LSTM(dataPath, bk_dataPath, modelName, niter=20):

        x_train, y_train = loadData(dataPath, bk_dataPath, shiftValue=USESHIFT, prob_add=False, seq_add=USESEQ)
    
        ## K-fold cross validation
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.DATABASE["rand_seed"])
        cvscores = []
    
        for train_idx, test_idx in kfold.split(x_train, y_train):
            
            y_train_1hot = to_categorical(y_train)

            model = models.Sequential()
            
            model.add(layers.Bidirectional(layers.LSTM(32)))
            model.add(layers.Dense(N_class, activation='softmax'))

            model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(x_train[train_idx], y_train_1hot[train_idx], epochs=EPOCH, batch_size=128, verbose=VB, validation_split=0.1, callbacks=CB)

            # evaluate the model
            t = model.predict(x_train[test_idx])
            t = np.argmax(t, axis=1)   
            acc=accuracy_score(y_train[test_idx], t)
            fscore = f1_score(y_train[test_idx], t, average="micro")
            print("*f1=%.4f, acc=%.2f" %(fscore,acc*100))
            cvscores.append(fscore)
            print confusion_matrix(y_train[test_idx], t) 

        print("%.4f (+/- %.4f)" % (np.mean(cvscores), np.std(cvscores)))


"""
Test set evluaiton functions. 
"""


def testEval_multi(model, x_test, y_test, rgs_test=None):

    logger.info("**********Evaluation on the testset*************")
    t = model.predict(x_test)
    t = np.argmax(t, axis=1)   

    print("Prediciton accuracy %f" %(accuracy_score(y_test, t)))
    print("F-score of multiple classification:")
    print(f1_score(y_test, t,average=None))
    print("Confuction matrix of the prediciton is: \n")
    print confusion_matrix(y_test, t) 


def testEval_roc(model, x_test, y_test, rgs_test=None):

    logger.info("**********Evaluation on the testset*************")
    t = model.predict(x_test).ravel()
    
    #Plot ROCcurve 
    fpr, tpr, thresholds = roc_curve(y_test, t)
    auc_value = auc(fpr, tpr)
    print auc_value

    plt.figure(1)
    plt.plot([0,1], [0,1], 'k--')
    plt.plot(fpr, tpr)
    plt.title('ROC curve')
    plt.savefig("../figures/ROC.png")


def testEval(model, x_test, y_test, rgs_test=None):

    logger.info("**********Evaluation on the testset*************")
    t = model.predict_classes(x_test)
    
    logger.info("* Total samples in testset %d" %(len(t)))
    
    TP,FP,TN,FN =0,0,0,0
    
    for i in range(x_test.shape[0]):
        
        gold = np.argmax(y_test[i])
        if gold == 1:
            testSet_output.write("[*bkp] %s:%s-%s\n" %(rgs_test[i][0], rgs_test[i][1], rgs_test[i][2]))

            if t[i] == gold:
                TP += 1
                if len(rgs_test) > 0:
                    correct_output.write("TP-region [%s, %s, %s]\n" %(rgs_test[i][0], rgs_test[i][1], rgs_test[i][2]))
            else:
                FN += 1
                if len(rgs_test) > 0:
                    error_output.write("FN-region [%s, %s, %s]\n" %(rgs_test[i][0], rgs_test[i][1], rgs_test[i][2]))
        else:
            testSet_output.write("[bg]  %s:%s-%s\n" %(rgs_test[i][0], rgs_test[i][1], rgs_test[i][2]))
            
            if t[i] == gold:
                TN += 1
            else:
                FP += 1
                if len(rgs_test) > 0:
                    error_output.write("FP-region [%s, %s, %s]\n" %(rgs_test[i][0], rgs_test[i][1], rgs_test[i][2]))
    
    logger.info("ACC=%f, TP=%d, TN=%d" %((TP+TN)/len(t), TP, TN))
    logger.info("F=%f" %(2*TP/(2*TP+FP+FN)))
    logger.info("TPR=%f" %((TP)/(TP+FN)))
    logger.info("FPR=%f" %((FP)/(FP+TN)))

    testSet_output.close()
    error_output.close()
    correct_output.close()
 
#################################################################################################
 ## Data visualization aim to explain why Training set is different from 180-CNV dataset.       
##################################################################################################

def dataVisual_PCA(dataPath, bk_dataPath, modelName, dataSetName):

        dim = 3

        x_data, y_data, rgs_data, x_cnv, y_cnv, rgs_cnv = loadData(dataPath, bk_dataPath, shiftValue=USESHIFT, prob_add=False, seq_add=USESEQ)

        y_data_label = np.apply_along_axis(checkLabel, 1, y_data)
        y_cnv_label = np.apply_along_axis(checkLabel, 1, y_cnv)
        
        x_data_pca = getPCAcomponent(x_data.reshape(x_data.shape[0], x_data.shape[1]), dim)
        x_cnv_pca = getPCAcomponent(x_cnv.reshape(x_cnv.shape[0], x_cnv.shape[1]), dim)

        fig = plt.figure()

        if USEPROB:
            figName = "../figures/PCA/Prob_"+ dataSetName + "_bin-"+ str(config.DATABASE["binSize"])+"_"+str(dim)+"D_data_PCA_visual.png"
        else:
            figName = "../figures/PCA/"+ dataSetName + "_bin-"+ str(config.DATABASE["binSize"])+"_"+str(dim)+"D_data_PCA_visual.png"
        
        index1 = [ idx for idx in range(len(y_data_label)) if y_data_label[idx] == 1] 
        index0 = [ idx for idx in range(len(y_data_label)) if y_data_label[idx] == 0]
        
        if dim == 2:
            print len(index1), len(index0)
            plt.scatter(x_data_pca[index1,0], x_data_pca[index1,1], color="blue") 
            plt.scatter(x_data_pca[index0,0], x_data_pca[index0,1], color="black")
            plt.scatter(x_cnv_pca[:,0], x_cnv_pca[:,1], color="red") 
            plt.legend(["train-1", "train-0", "CNV-1"],loc='upper left')
        
        if dim == 3:

            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x_data_pca[index1,0], x_data_pca[index1,1],  x_data_pca[index1,2], color="blue", marker='o')
            ax.scatter(x_data_pca[index0,0], x_data_pca[index0,1],  x_data_pca[index0,2], color="black", marker='x')
            ax.scatter(x_cnv_pca[:,0], x_cnv_pca[:,1], x_cnv_pca[:,2], color="red", marker='*')
            plt.legend(["train-1", "train-0", "CNV-1"], loc='upper left')
        
        plt.savefig(figName)

        # output the distance 
        x_data_1 = np.apply_along_axis(np.mean, 0, x_data[index1,:])
        print x_data_1.shape
        x_data_0 = np.apply_along_axis(np.mean, 0, x_data[index0,:])
        x_cnv_1 = np.apply_along_axis(np.mean, 0, x_cnv)
        
        print("CV distance %f" %(np.linalg.norm(x_data_0 - x_data_1)))
        print("CNV distance %f" %(np.linalg.norm(x_data_0 - x_cnv_1)))



def dataVisual_tSNE(dataPath, bk_dataPath, modelName, dataSetName):

        dim, niter = 2, 1000

        x_data, y_data, rgs_data, x_cnv, y_cnv, rgs_cnv = loadData(dataPath, bk_dataPath, shiftValue=USESHIFT, prob_add=USEPROB, seq_add=USESEQ)

        y_data_label = np.apply_along_axis(checkLabel, 1, y_data)
        y_cnv_label = np.apply_along_axis(checkLabel, 1, y_cnv)

        tsne = TSNE(n_components=dim, verbose=VB, perplexity=40, n_iter=niter)
        x_data_tsne = tsne.fit_transform(x_data.reshape(x_data.shape[0], x_data.shape[1]))
        x_cnv_tsne =  tsne.fit_transform(x_cnv.reshape(x_cnv.shape[0], x_cnv.shape[1]))

        fig = plt.figure()
        if USEPROB:
            figName = "../figures/tSNE/Prob_"+ dataSetName + "_bin-"+ str(config.DATABASE["binSize"])+"_"+str(dim)+"D_data_tSNE_visual.png"
        else:
            figName = "../figures/tSNE/"+ dataSetName + "_bin-"+ str(config.DATABASE["binSize"])+"_"+str(dim)+"D_data_tSNE_visual.png"
        
        index1 = [ idx for idx in range(len(y_data_label)) if y_data_label[idx] == 1] 
        index0 = [ idx for idx in range(len(y_data_label)) if y_data_label[idx] == 0]

        if dim == 2:
            print len(index1), len(index0)
            plt.scatter(x_data_tsne[index1,0], x_data_tsne[index1,1], color="blue") 
            plt.scatter(x_data_tsne[index0,0], x_data_tsne[index0,1], color="black")
            plt.scatter(x_cnv_tsne[:,0], x_cnv_tsne[:,1], color="red") 
            plt.legend(["train-1", "train-0", "CNV-1"], loc='upper left')
            plt.savefig(figName)

        if dim == 3:

            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x_data_tsne[index1,0], x_data_tsne[index1,1],  x_data_tsne[index1,2], color="blue")
            ax.scatter(x_data_tsne[index0,0], x_data_tsne[index0,1],  x_data_tsne[index0,2], color="black")
            ax.scatter(x_cnv_tsne[:,0], x_cnv_tsne[:,1], x_cnv_tsne[:,2], color="red")
            plt.legend(["train-1", "train-0", "CNV-1"], loc='upper left')
            plt.savefig(figName)

        # output the distance 
        x_data_1 = np.apply_along_axis(np.mean, 0, x_data[index1,:])
        x_data_0 = np.apply_along_axis(np.mean, 0, x_data[index0,:])
        x_cnv_1 = np.apply_along_axis(np.mean, 0, x_cnv)
        
        print("CV distance %f" %(np.linalg.norm(x_data_0 - x_data_1)))
        print("CNV distance %f" %(np.linalg.norm(x_data_0 - x_cnv_1)))





