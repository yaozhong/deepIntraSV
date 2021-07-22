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
from datetime import date


TRAIL=100
EPOCH=30

# model training settings
CB = [ callbacks.EarlyStopping(monitor="val_loss", patience=10, mode = "auto", restore_best_weights=True) ] 
VB = 0

## 2019/12/19 add ROC curve for the figure
def plot_roc(gold_cnv, pred_cnv, rocFigPath):

    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc_value)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    plt.savefig(rocFigPath)


def binary_eval(gold_cnv, pred_cnv, modelInfo, binaryCase=False, rocFigPath=""):

        if binaryCase == False:
            gold_label = np.apply_along_axis(checkLabel, 1, gold_cnv)
            pred_label = np.apply_along_axis(checkLabel, 1, pred_cnv)

        else:
            gold_label = gold_cnv
            pred_label = pred_cnv

        # based on the gold standard split
        index1 = [ idx for idx in range(gold_label.shape[0]) if gold_label[idx] == 1]
        index0 = [ idx for idx in range(gold_label.shape[0]) if gold_label[idx] == 0]

        print("* [Data_INFO]: Test BK=%d / Total=%d" %(len(index1), len(gold_label)))
        print "* [Model_INFO]:" + modelInfo 

        # dice score section
        if binaryCase == False:
            print "* [Segmentation Results]:"
            df_cnv = dice_score(gold_cnv, pred_cnv)
            df1 = dice_score(gold_cnv[index1], pred_cnv[index1])  

            print("All_dice=%.4f" %(df_cnv)),
            print(", BK_dice=%.4f" %(df1)),
            
            iou_cnv = iou_score(gold_cnv, pred_cnv)
            iou1 = iou_score(gold_cnv[index1], pred_cnv[index1])

            if len(index0) > 0:
                df0 = dice_score(gold_cnv[index0], pred_cnv[index0])   
                iou0 = iou_score(gold_cnv[index0], pred_cnv[index0])        
            else:
                df0, iou0 = None, None

        print "* [Binary classification Results]:"
        fscore = f1_score(gold_label, pred_label, average="micro")
        print ("F-score=%f" %(fscore)),
        tn, fp, fn, tp = confusion_matrix(gold_label, pred_label).ravel()   
        auc_value = None    

        if len(index0) > 0 :
            fpr, tpr, thresholds = roc_curve(gold_label, pred_label)

            print(", Sensitivity(TPR)=%f" %(tp/(tp+fn))),
            print(", Specificity(TNR)=%f" %(tn/(tn+fp))),
            print(", FDR=%f" %(fp/(fp+tp)))

            print("* Precision=%f" %(tp/(tp+fp))),
            print(", Recall=%f" %(tp/(tp+fn)))
            
        else:
            auc_value = None


        print("* [Confucation Matrix]:")
        print confusion_matrix(gold_label, pred_label) 
        
        if binaryCase == True:
            return (fscore, auc_value, tp/(tp+fn), fp/(fp+tp), tp/(tp+fp), tp/(tp+fn))
        else:
            return (df_cnv, df1, df0, fscore, auc_value, tp/(tp+fn), fp/(fp+tp), iou_cnv, iou1, iou0, tp/(tp+fp), tp/(tp+fn))


###########################################################
### evluation of the break point, basic evluation metric
###########################################################
def evluation_breakpoint(x_cnv, rgs_cnv, gold_cnv, pred_cnv, figureSavePath, plotFig=False):
 
    same_bk, diff1_bk, more2_diff = [], [], []

    # current evluation is not the final one.
    basic_distance = []

    for i in range(gold_cnv.shape[0]):
        pred_seq, pred_ident_count, pred_seq_point = get_break_point_position(pred_cnv[i])
        gold_seq, gold_ident_count, gold_seq_point = get_break_point_position(gold_cnv[i])

        # based on the gold standard split
        if np.abs(len(pred_ident_count) - len(gold_ident_count)) >= 2:
            more2_diff.append(i)

        elif len(pred_ident_count) == len(gold_ident_count):
            same_bk.append(i)
            
            # partial evluation of the predicited distance
            if len(pred_seq) > 1:
                basic_distance.extend(np.abs(pred_seq_point[:-1] - gold_seq_point[:-1]))
        else:
            diff1_bk.append(i)
     
            """
            print(pred_seq),
            print(pred_seq_point)

            print(gold_seq),
            print(gold_seq_point)

            print(rgs_cnv[i])
            print("-"*10)
            """

    print("** Totally have [%d] same bk, [%d] 1-diff bbk, [%d] 2 more-diff"  \
        %(len(same_bk), len(diff1_bk), len(more2_diff)))

    if len(basic_distance) > 0:
        print("** Equal length prediciotn has the Breakpoint prediction shift of [%d, %d]" %(np.mean(basic_distance), np.std(basic_distance)))

    # visualization
    if plotFig:
        if len(same_bk) > 0:
            visual_prediction(x_cnv[same_bk], rgs_cnv[same_bk], gold_cnv[same_bk], pred_cnv[same_bk], figureSavePath + "_same")
        if len(diff1_bk) > 0:
            visual_prediction(x_cnv[diff1_bk], rgs_cnv[diff1_bk], gold_cnv[diff1_bk], pred_cnv[diff1_bk], figureSavePath + "_1diff")
        if len(more2_diff) > 0:
            visual_prediction(x_cnv[more2_diff], rgs_cnv[more2_diff], gold_cnv[more2_diff], pred_cnv[more2_diff], figureSavePath + "_2moreDiff")
    



###############################################################################
# paper description break point enhancement
###############################################################################
def get_new_breakpoint2(x_cnv, rgs_cnv, gold_cnv, pred_cnv, figureSavePath, plotFig=False):
 
    # current evluation is not the final one.
    basic_distance = []
    num_1, num_0, num_more = 0, 0, 0
    new_bks = []
    # this one is used to prompt potential false positive.
    non_segments = []

    for i in range(gold_cnv.shape[0]):
        pred_seq, pred_ident_count, pred_seq_point = get_break_point_position(pred_cnv[i])
        gold_seq, gold_ident_count, gold_seq_point = get_break_point_position(gold_cnv[i])

        old_bk = rgs_cnv[i][1]+ int(config.DATABASE["binSize"]/2)

        # need to revised, SV region segment
        if len(pred_seq) == 2:
            new_bk = rgs_cnv[i][1]+ pred_seq_point[0]
            new_bks.append((rgs_cnv[i][0], old_bk, new_bk, rgs_cnv[i][3]))
            num_1 += 1

        if len(pred_seq) == 1:
            # makes no prediction of the current ones
            new_bks.append((rgs_cnv[i][0], old_bk, old_bk, rgs_cnv[i][3]))
            num_0 += 1

        if len(pred_seq) > 2:
            new_bk = rgs_cnv[i][1]+ pred_seq_point[0]
            min_dist = np.abs(old_bk - new_bk)

            for j in range(1, len(pred_seq)):
                cur_bk = rgs_cnv[i][1]+ pred_seq_point[j]
                # fix the bug of distance calcuation
                cur_dist = np.abs(old_bk - cur_bk)
                if(cur_dist < min_dist):
                    min_dist = cur_dist
                    new_bk = cur_bk

            new_bks.append((rgs_cnv[i][0], old_bk, new_bk, rgs_cnv[i][3]))
            num_more += 1

    print("[*] Enhancement Changes: no shift is %f, single_pred shift %f,  more_pred shift %f, expected total %d" %(num_0/gold_cnv.shape[0], num_1/gold_cnv.shape[0], num_more/gold_cnv.shape[0], gold_cnv.shape[0]))

    # generate regions for old_bk and new_bk
    old_rg_list, new_rg_list = [],[]
    half = int(len(new_bks)/2)
    for i in range(half):
        old_rg_list.append((new_bks[i][0], new_bks[i][1], new_bks[i+half][1], new_bks[i][3], 0, 0, 0, 0))
        new_rg_list.append((new_bks[i][0], new_bks[i][2], new_bks[i+half][2], new_bks[i][3], 0, 0, 0, 0))

    return old_rg_list, new_rg_list


# more agreesive enhancement
def get_new_breakpoint(x_cnv, rgs_cnv, gold_cnv, pred_cnv, figureSavePath, plotFig=False):
 
    # current evluation is not the final one.
    basic_distance = []
    num_same, num_less, num_over = 0, 0, 0
    new_bks = []

    for i in range(gold_cnv.shape[0]):
        pred_seq, pred_ident_count, pred_seq_point = get_break_point_position(pred_cnv[i])
        gold_seq, gold_ident_count, gold_seq_point = get_break_point_position(gold_cnv[i])

        old_bk = rgs_cnv[i][1]+ int(config.DATABASE["binSize"]/2)

        # ideal improvment case
        if len(pred_seq) == 2:
            num_same += 1

            new_bk = rgs_cnv[i][1]+ pred_seq_point[0]
            new_bks.append((rgs_cnv[i][0], old_bk, new_bk, rgs_cnv[i][3]))

        # less prediction
        if len(pred_seq) == 1:
            num_less += 1
            new_bks.append((rgs_cnv[i][0], old_bk, old_bk, rgs_cnv[i][3]))

        # more prediction
        if len(pred_seq) > 2:
            num_over += 1
            new_bks.append((rgs_cnv[i][0], old_bk, old_bk, rgs_cnv[i][3]))

    print("[*] Ideal segment number is %f, over %f, less %f, expected total %d" %(num_same/gold_cnv.shape[0], num_over/gold_cnv.shape[0], num_less/gold_cnv.shape[0], gold_cnv.shape[0]))

    # generate regions for old_bk and new_bk
    old_rg_list, new_rg_list = [],[]
    half = int(len(new_bks)/2)
    for i in range(half):
        old_rg_list.append((new_bks[i][0], new_bks[i][1], new_bks[i+half][1], new_bks[i][3], 0, 0, 0, 0))
        new_rg_list.append((new_bks[i][0], new_bks[i][2], new_bks[i+half][2], new_bks[i][3], 0, 0, 0, 0))

    return old_rg_list, new_rg_list

"""
1-vs-1 SVM model
"""
def SVM_CV(dataPath, bk_dataPath, modelParamPath, modelSavePath, dataInfo, plotResult= False, plotTrainCurve=True):

        # model parameters
        modelInfo = "SVM-CV"

        x_data, y_data, rgs_data, bps_data, x_cnv, y_cnv, rgs_cnv, bps_cnv = loadData(dataPath, bk_dataPath, \
            prob_add=config.DATABASE["USEPROB"], seq_add=config.DATABASE["USESEQ"], gc_norm=config.DATABASE["GC_NORM"])

        x_data = np.concatenate((x_data, x_cnv), 0)
        y_data = np.concatenate((y_data, y_cnv), 0)
        rgs_data = np.concatenate((rgs_data, rgs_cnv), 0)
        print("After concatenating ...")
        print(x_data.shape)

        y_data = y_data.reshape(y_data.shape[0], y_data.shape[1], 1)
        y_data_label = np.apply_along_axis(checkLabel, 1, y_data)
        y_data_label = y_data_label.reshape(y_data_label.shape[0])
        
        x_data = x_data.reshape(x_data.shape[0], x_data.shape[1])
        y_data = y_data_label

        ## K-fold cross validation
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.DATABASE["rand_seed"])
        cv_scores, cv_bk_scores, cv_bg_scores, cv_auc, cnv_scores= [], [], [], [], []
        cv_sensitivity, cv_FDR = [], []
        cv_precision, cv_recall = [], []

        index = 0
        for train_idx, test_idx in kfold.split(x_data, y_data_label): 

            index = index + 1

            if config.DATABASE["small_cv_train"] == "small":
                print("+ Small train CV actived!!!! ")
                train_idx, test_idx = test_idx, train_idx

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

            fscore, auc_value, sensitivity, FDR, precision, recall = binary_eval(gold, pred, modelInfo, True)

            cv_FDR.append(FDR)
            cv_precision.append(precision)
            cv_recall.append(recall)

        ####################################  Generating Results Report #################################
        
        print "\n=========== [DATA/MODEL information] ============="
        print "[-CV-" + dataInfo+ "]"
        print "["+ modelSavePath + "]"
        print "["+ modelInfo + "]"

        print '-'*30
        print("-- CV Precision %.4f (%.4f)" % (np.mean(cv_precision), np.std(cv_precision)))
        print("-- CV Recall %.4f (%.4f)" % (np.mean(cv_recall), np.std(cv_recall)))
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



def SVM(dataPath, bk_dataPath, modelParamPath, dataInfo, modelInfo, ):

        x_data, y_data, rgs_data, bps_data, x_cnv, y_cnv, rgs_cnv, bps_cnv = loadData(dataPath, bk_dataPath, \
            prob_add=config.DATABASE["USEPROB"], seq_add=config.DATABASE["USESEQ"], gc_norm=config.DATABASE["GC_NORM"])

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
    

        model = svm.SVC()  # 1-vs-1
        model.fit(x_train, y_train) 
        t = model.predict(x_test).ravel()

        print "\n=========== [DATA/MODEL information] ============="
        print "[" + dataInfo+ "]"
        print("Train BK=%d / Total=%d" %(np.sum(y_data_label), y_data_label.shape[0]))
        
        figureSavePath= "../experiment/ROC/" + date.today().strftime("%Y%m%d") + "_"
        binary_eval(y_test, t, modelInfo, True, figureSavePath + "SVM.png")


# added as one-class SVM
def SVM_1class(dataPath, bk_dataPath, modelParamPath, dataInfo, modelInfo):

        print("* Training and evluation of OneClass SVM ...")

        x_data, y_data, rgs_data, bps_data, x_cnv, y_cnv, rgs_cnv, bps_cnv = loadData(dataPath, bk_dataPath, \
         prob_add=config.DATABASE["USEPROB"], seq_add=config.DATABASE["USESEQ"], gc_norm=config.DATABASE["GC_NORM"])

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

        # only keep SV regions for classify
        print("filtering the target index ... for normal")
        target_index = [ idx for idx in range(len(y_data_label)) if y_data_label[idx] == 0]
        x_train = x_train[target_index]

        model = svm.OneClassSVM(gamma='auto')
        model.fit(x_train) 
        t = model.predict(x_test).ravel()
       
        # the outline is -1, change to the 0
        t = [0 if x== 1 else x for x in t]
        t = [1 if x==-1 else x for x in t]
        
        print "\n=========== [DATA/MODEL information] ============="
        print "[" + dataInfo+ "]"
        print("Train BK=%d / Total=%d" %(np.sum(y_data_label), y_data_label.shape[0]))
        
        figureSavePath= "../experiment/ROC/" + date.today().strftime("%Y%m%d") + "_"
        binary_eval(y_test, t, modelInfo, True, figureSavePath + "SVM_1class_trainNorm.png")


# classical CNN
def CNN_networkstructure(kernel_size, window_len, maxpooling_len, BN=True, DropoutRate=0.2):

        model = models.Sequential()
        # 1000bp (32,10,10) -> ()
        model.add(layers.Conv1D(kernel_size[0], window_len[0], activation="relu"))
        model.add(layers.MaxPooling1D(maxpooling_len[0]))
        
        model.add(layers.Conv1D(kernel_size[1], window_len[1],activation="relu"))
        model.add(layers.MaxPooling1D(maxpooling_len[1]))

        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dropout(DropoutRate))
        model.add(layers.Dense(config.DATABASE["binSize"], activation='sigmoid'))
    
        return model

# 2019-01-04 add the CNN version for evluation
def CNN(dataPath, bk_dataPath, modelParamPath, modelSavePath, dataInfo, plotTrainCurve=False, plotResult=False):

        # Data loading start
        x_data, y_data, rgs_data, bps_data, x_cnv, y_cnv, rgs_cnv, bps_cnv = loadData(dataPath, bk_dataPath,  \
            prob_add=config.DATABASE["USEPROB"], seq_add=config.DATABASE["USESEQ"], gc_norm=config.DATABASE["GC_NORM"])

        # for the convolutional output
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

        if config.DATABASE["USEPROB"]: modelInfo += "_USEPROB"
        if config.DATABASE["USESEQ"]: modelInfo += "_USESEQ"
        if config.DATABASE["GC_NORM"]: modelInfo += "_GC-NORM"

        #########################################
        modelSaveName = os.path.basename(modelParamPath) + config.DATABASE["model_data_tag"]  #+ "|" +modelInfo

        rd_input = Input(shape=(x_data.shape[1], x_data.shape[-1]), dtype='float32', name="rd")
        model = CNN_networkstructure(kernel_size, window_len, maxpooling_len, BN, dropoutRate)
        model.compile(optimizer=Adam(lr = lr) , loss= dice_loss, metrics=[dice_coef])
        history = model.fit(x_data, y_data, epochs=params["epoch"], batch_size=batchSize, verbose=VB \
        ,callbacks=CB, validation_split=0.2)
        
        print "@ Saving model ..."
        model.save(modelSavePath + modelSaveName +".h5" )
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

        ############# dice evluation #################
        t_cnv = model.predict(x_cnv, verbose=VB)
        pred_cnv = (t_cnv > 0.5).astype(np.float32).reshape(t_cnv.shape[0], t_cnv.shape[1])
        gold_cnv = y_cnv.astype(np.float32)

        # save the prediction figures
        if plotResult == True:
            figureSavePath= "../experiment/result/"+ modelInfo + "_DATA-"+dataInfo
            visual_prediction(x_cnv, rgs_cnv, gold_cnv, pred_cnv, figureSavePath)

        figureSavePath= "../experiment/result/" + date.today().strftime("%Y%m%d") + "_"
        evluation_breakpoint(x_cnv, rgs_cnv, gold_cnv, pred_cnv, figureSavePath+"CNN_", False)

        print "\n=========== [DATA/MODEL information] ============="
        print "[" + dataInfo+ "]"
        print("Train BK=%d / Total=%d" %(np.sum(y_data_label), y_data_label.shape[0]))

        figureSavePath= "../experiment/ROC/" + date.today().strftime("%Y%m%d") + "_"
        print(t_cnv.shape)
        binary_eval(gold_cnv, pred_cnv, modelInfo, False, figureSavePath + "_CNN.png")


def CNN_CV(dataPath, bk_dataPath, modelParamPath, modelSavePath, dataInfo, plotResult= False, plotTrainCurve=False):

        # all data loading
        x_data, y_data, rgs_data, bps_data, x_cnv, y_cnv, rgs_cnv, bps_cnv = loadData(dataPath, bk_dataPath, \
            prob_add=config.DATABASE["USEPROB"], seq_add=config.DATABASE["USESEQ"], gc_norm=config.DATABASE["GC_NORM"])

        # 2019-12-05 added, previous version the CV part is on the split-train set only
        # concatenate 
        x_data = np.concatenate((x_data, x_cnv), 0)
        y_data = np.concatenate((y_data, y_cnv), 0)
        rgs_data = np.concatenate((rgs_data, rgs_cnv), 0)
        print("After concatenating ...")
        print(x_data.shape)

        # checking comment
        #y_data = y_data.reshape(y_data.shape[0], y_data.shape[1], 1)
        y_data_label = np.apply_along_axis(checkLabel, 1, y_data)
        print("BK=%d / Total=%d" %(np.sum(y_data_label), y_data_label.shape[0]))
 
        # model parameters loading
        if config.DATABASE["model_param"] != "":
            params = load_modelParam(config.DATABASE["model_param"])
            print "*"*40
            print params
            print "*"*40
        else:
            print("[Error!] model parameter file is not provided, please check!")
            exit(-1)

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

        if config.DATABASE["USEPROB"]: modelInfo += "_USEPROB"
        if config.DATABASE["USESEQ"]: modelInfo += "_USESEQ"
        if config.DATABASE["GC_NORM"]: modelInfo += "_GC-NORM"

        ## K-fold cross validation
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.DATABASE["rand_seed"])
        cv_scores, cv_bk_scores, cv_bg_scores, cv_auc, cnv_scores= [], [], [], [], []
        cv_iou, cv_bk_iou, cv_bg_iou = [],[],[]
        cv_sensitivity, cv_FDR = [], []
        cv_precision, cv_recall = [], []

        index = 0
        for train_idx, test_idx in kfold.split(x_data, y_data_label): 

            index = index + 1

            if config.DATABASE["small_cv_train"] == "small":
                print("+ Small train CV actived!!!! ")
                train_idx, test_idx = test_idx, train_idx

            rd_input = Input(shape=(x_data.shape[1], x_data.shape[-1]), dtype='float32', name="rd")
            
            model = CNN_networkstructure(kernel_size, window_len, maxpooling_len, BN, dropoutRate)
            model.compile(optimizer=Adam(lr = lr) , loss= dice_loss, metrics=[dice_coef])
            
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

            # Break point evluation
            figureSavePath= "../experiment/result/" + date.today().strftime("%Y%m%d") + "_"
            evluation_breakpoint(x_data[test_idx], rgs_data[test_idx], gold, pred, figureSavePath+"CNN_CV_", False)

            df, df1, df0, fscore, auc_value, sensitivity, FDR, iou,iou1,iou0, precision, recall = binary_eval(gold, pred, modelInfo, False)

            cv_scores.append(df)
            cv_bk_scores.append(df1)
            cv_bg_scores.append(df0)

            cv_iou.append(iou)
            cv_bk_iou.append(iou1)
            cv_bg_iou.append(iou0)
            
            cv_FDR.append(FDR)

            cv_precision.append(precision)
            cv_recall.append(recall)

        ####################################  Generating Results Report #################################     
        print "\n=========== [DATA/MODEL information] ============="
        print "[-CV-" + dataInfo+ "]"
        print "["+ modelSavePath + "]"
        print "["+ modelInfo + "]"

        print '-'*30
        print("-- CV Precision %.4f (%.4f)" % (np.mean(cv_precision), np.std(cv_precision)))
        print("-- CV Recall %.4f (%.4f)" % (np.mean(cv_recall), np.std(cv_recall)))
        print("-- CV FDR %.4f (%.4f)" % (np.mean(cv_FDR), np.std(cv_FDR)))
        print '*'*30

        print '-'*30
        print("* CV all dice %.4f (%.4f)" % (np.mean(cv_scores), np.std(cv_scores)))
        print("-- CV BK dice %.4f (%.4f)" % (np.mean(cv_bk_scores), np.std(cv_bk_scores)))
        print("-- CV BG dice %.4f (%.4f)" % (np.mean(cv_bg_scores), np.std(cv_bg_scores)))
        print '-'*30

        print '-'*30
        print("* CV all IOU %.4f (%.4f)" % (np.mean(cv_iou), np.std(cv_iou)))
        print("-- CV BK IOU %.4f (%.4f)" % (np.mean(cv_bk_iou), np.std(cv_bk_iou)))
        print("-- CV BG IOU %.4f (%.4f)" % (np.mean(cv_bg_iou), np.std(cv_bg_iou)))




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





