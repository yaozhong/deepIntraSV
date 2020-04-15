"""
Date: 2018-08-16
Author: Yao-zhong Zhang
Description: the basic training for the keras model for the break point detection problem.

2018-12-08:Model concrete test for segmentation.
"""
# -*- coding: utf-8 -*-

from __future__ import division

from util import *
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
from models.model_unet import *
from models.model_baseline import *
from models.model_hyperOpt import *

from dataProcess.data_cacheLoading import *

#from train import *

# model training settings
CB = [ callbacks.EarlyStopping(monitor="val_dice_coef", patience=20, restore_best_weights=True) ] 
VB = 1

# how to get the segment, only return the first 1 position, not very accurate
def getSegPosition(pred_cnv):

    # no-segment predicted
    if np.sum(pred_cnv) == 0:
        return(-1)

    # the first of the segment
    for i in range(len(pred_cnv)):
        if pred_cnv[i] == 1:
            return(i)

# might be a good evluation metric in the current stage
def position_eval(gold_cnv, pred_cnv, rgs_cnv):

    eval_dic = {}
    num_ci, num_non_ci = 0,0
    num_in_ci, num_in_ci100, num_in_ci50, num_in_ci10 = 0,0,0,0
    dist_list = []

    for i in range(len(rgs_cnv)):

        svType = rgs_cnv[i][3]
        left_or_right = rgs_cnv[i][4]
        
        pred_pos = getSegPosition(pred_cnv[i])
        gold_pos = int(rgs_cnv[i][5])
        ci_left = int(rgs_cnv[i][6])
        ci_right = int(rgs_cnv[i][7])

        
        if rgs_cnv[i][3] == "BG":
            continue

        dist_list.append(np.abs(pred_pos-gold_pos))

        # no confidence interval information
        if ci_left > 0 and ci_right < 0:
            num_non_ci += 1
            continue
        

        if ci_right - ci_left < 100:
            print("Predict=%d, gold_pos Range=[%d, %d]" %(pred_pos, gold_pos+ci_left, gold_pos+ci_right))

        num_ci += 1

        if pred_pos >= gold_pos + ci_left and pred_pos <= gold_pos + ci_right:
            num_in_ci += 1

    print("In-ci segment is %d/%d" %(num_in_ci, num_ci))
    print("non-ci BK sample is %d" %(num_non_ci))
    print np.mean(dist_list), np.min(dist_list),np.max(dist_list)


def label_eval(gold_cnv, pred_cnv, rgs_cnv):
    
    eval_dic = {}
    for i in range(len(rgs_cnv)):
        svType = rgs_cnv[i][3]
        if svType not in eval_dic.keys():
            eval_dic[svType] = [0,0,0,0]

        if (np.sum(gold_cnv[i]) > 0) == (np.sum(pred_cnv[i])>0):
            eval_dic[svType][0] +=1
            eval_dic[svType][3] += dice_score(gold_cnv[i], pred_cnv[i])
        
        eval_dic[svType][1] +=1
        eval_dic[svType][2] += dice_score(gold_cnv[i], pred_cnv[i])


    for t in eval_dic.keys():
        if eval_dic[t][0] > 0:
            print("[%s]:\t* binary= %d/%d,\t%f,\tAvg_Dice-all=%f\tAvg_Dice-BK=%f"  \
                    %(t, eval_dic[t][0], eval_dic[t][1], eval_dic[t][0]/eval_dic[t][1], \
                    eval_dic[t][2]/eval_dic[t][1],  eval_dic[t][3]/eval_dic[t][0] ))
        else:
            print("[%s]:\t* binary= %d/%d,\t%f,\tAvg_Dice-all=%f,\tAve_Dice-BK=NULL"  \
                    %(t, eval_dic[t][0], eval_dic[t][1], eval_dic[t][0]/eval_dic[t][1], \
                    eval_dic[t][2]/eval_dic[t][1]))



def UNet(dataPath, bk_dataPath, modelSavePath, dataInfo, plotResult=False):

        
        # Data loading start
        x_data, y_data, rgs_data, x_cnv, y_cnv, rgs_cnv = loadData(dataPath, bk_dataPath, prob_add=USEPROB, seq_add=USESEQ)

        # for the convolutional output
        y_data = y_data.reshape(y_data.shape[0], y_data.shape[1], 1)
        y_data_label = np.apply_along_axis(checkLabel, 1, y_data)
        print("BK=%d / Total=%d" %(np.sum(y_data_label), y_data_label.shape[0]))       

        #########################################
        
        # loading model 
        print("* Loading model parameters...")
        model = models.load_model(modelSavePath, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef':dice_coef})


        ###########################################################################################3
        # Results generation
        ###########################################################################################

        ############# dice evluation #################
        t_cnv = model.predict(x_cnv, verbose=VB)
        pred_cnv = (t_cnv > 0.5).astype(np.float32).reshape(t_cnv.shape[0], t_cnv.shape[1])
        gold_cnv = y_cnv.astype(np.float32)

        # save the prediction figures
        if plotResult == True:
            figureSavePath= "../experiment/visual/" + "_DATA-"+dataInfo

            visual_prediction(x_cnv, rgs_cnv, gold_cnv, pred_cnv, figureSavePath)


        print "\n=========== [DATA/MODEL information] ============="
        print "[" + dataInfo+ "]"
        print("Train BK=%d / Total=%d" %(np.sum(y_data_label), y_data_label.shape[0]))


        binary_eval(gold_cnv, pred_cnv, "UNet_test"+ os.path.basename(modelSavePath))
        ###############################################
        label_eval(gold_cnv, pred_cnv, rgs_cnv)
        #position_eval(gold_cnv, pred_cnv, rgs_cnv)



def CNN(dataPath, bk_dataPath, modelSavePath, dataInfo, plotResult=False):

        
        # Data loading start
        x_data, y_data, rgs_data, x_cnv, y_cnv, rgs_cnv = loadData(dataPath, bk_dataPath, prob_add=USEPROB, seq_add=USESEQ)

        # for the convolutional output
        y_data = y_data.reshape(y_data.shape[0], y_data.shape[1], 1)
        y_data_label = np.apply_along_axis(checkLabel, 1, y_data)
        print("BK=%d / Total=%d" %(np.sum(y_data_label), y_data_label.shape[0]))       

        #########################################
        
        # loading model 
        print("* Loading model parameters...")
        model = models.load_model(modelSavePath, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef':dice_coef})

        ###########################################################################################3
        # Results generation
        ###########################################################################################

        ############# dice evluation #################
        t_cnv = model.predict(x_cnv, verbose=VB)
        pred_cnv = (t_cnv > 0.5).astype(np.float32).reshape(t_cnv.shape[0], t_cnv.shape[1])
        gold_cnv = y_cnv.astype(np.float32)

        # save the prediction figures
        if plotResult == True:
            figureSavePath= "../experiment/visual/" + "_DATA-"+dataInfo
            visual_prediction(x_cnv, rgs_cnv, gold_cnv, pred_cnv, figureSavePath)


        print "\n=========== [DATA/MODEL information] ============="
        print "[" + dataInfo+ "]"
        print("Train BK=%d / Total=%d" %(np.sum(y_data_label), y_data_label.shape[0]))

        binary_eval(gold_cnv, pred_cnv, "CNN:w_test"+ os.path.basename(modelSavePath))
        ###############################################
        label_eval(gold_cnv, pred_cnv, rgs_cnv)
        #position_eval(gold_cnv, pred_cnv, rgs_cnv)


if  __name__ == "__main__":
    
        parser = argparse.ArgumentParser(description='Intra-bin SV segmentation for base-pair read-depth signals')
        parser.add_argument('--gpu', '-g', type=str, default="3", help='GPU ID for Training model.')
        parser.add_argument('--bin', '-b', type=int, default=1000, required=True, help='bin size of target region')
        parser.add_argument('--dataAug', '-da', type=int, default=0, help='Epochs of data augmentation')
        parser.add_argument('--model', '-m', type=str, default="UNet", required=True, help='Model names')
        
        parser.add_argument('--dataSplit', '-ds', type=str, default="RandRgs",help='Data split setting')
        parser.add_argument('--evalMode', '-em', type=str, default="single", required=True, help='Data evaluation mode')
        
        parser.add_argument('--dataSelect', '-d', type=str, default="na12878_60x", required=True, help='bam file')
        parser.add_argument('--dataSelect2', '-d2', type=str, default="", help='bam file for cross-sample testing.')

        parser.add_argument('--modelParam', '-mp', type=str, default="", help='reload pre-determined model hyper-parameters')       
        parser.add_argument('--modelWeight', '-mw', type=str, default="", help='resign pre-determined model weight')

        # update accroding to the latest parmaters
        args = parser.parse_args()
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        if args.bin != config.DATABASE["binSize"]:
            config.DATABASE["binSize"] = args.bin

        if args.dataAug != config.DATABASE["data_aug"]:
            config.DATABASE["data_aug"] = args.dataAug
            
        if args.evalMode != config.DATABASE["eval_mode"]:
            config.DATABASE["eval_mode"] = args.evalMode

        if args.dataSplit != config.DATABASE["data_split"]:
            config.DATABASE["data_split"] = args.dataSplit
    
        if args.modelParam != "":
            config.DATABASE["model_param"] = args.modelParam

        binSize = config.DATABASE["binSize"]

        ANNOTAG=""

        ## background ata first caching check
        bk_dataPath = "../data/data_cache/"+ ANNOTAG + args.dataSelect \
                +"_"+config.DATABASE["count_type"] \
                +"_bin"+str(binSize)+"_GENOMESTAT_"
                
        bk_dataPath += "SampleRate-" + str(config.DATABASE["genomeSampleRate"]) 
        bk_dataPath += "_Filter-Mappability-"+ str(config.DATABASE["mappability_threshold"])

        bamFilePath = config.BAMFILE[args.dataSelect]

        if not os.path.exists(bk_dataPath):
            cache_genome_statistics(bamFilePath, bk_dataPath, config.DATABASE["genomeSampleRate"])


        ########### Prepare the second genome for normalization ###########
        if(config.DATABASE["eval_mode"]=="cross"):

            bk_dataPath2 = "../data/data_cache/"+ ANNOTAG + args.dataSelect2 \
                    +"_"+config.DATABASE["count_type"] \
                    +"_bin"+str(binSize)+"_GENOMESTAT_"
                
            bk_dataPath2 += "SampleRate-" + str(config.DATABASE["genomeSampleRate"]) 
            bk_dataPath2 += "_Filter-Mappability-"+ str(config.DATABASE["mappability_threshold"])
            
            bamFilePath2 = config.BAMFILE[args.dataSelect2]

            if not os.path.exists(bk_dataPath2):
                cache_genome_statistics(bamFilePath2, bk_dataPath2, config.DATABASE["genomeSampleRate"])
            
            bk_dataPath = [bk_dataPath, bk_dataPath2]
        else:
            bk_dataPath = [bk_dataPath]

        ## training data first cachcing check
        if config.DATABASE["eval_mode"] == "cross":
            assert(args.dataSelect2 != "")
            goldFile = [config.AnnoCNVFile[args.dataSelect], config.AnnoCNVFile[args.dataSelect2]]
            bamFilePath = [config.BAMFILE[args.dataSelect], config.BAMFILE[args.dataSelect2]]
        else:
            goldFile = [config.AnnoCNVFile[args.dataSelect]]
            bamFilePath = [config.BAMFILE[args.dataSelect]]


        dataPath = "../data/data_cache/"+ ANNOTAG         
        dataInfo = args.dataSelect +"_"+config.DATABASE["count_type"] +"_bin"+str(binSize)+"_TRAIN"
        dataInfo += "_extendContext-" + str(config.DATABASE["extend_context"])
        dataInfo += "_dataSplit-" + config.DATABASE["data_split"] 
        dataInfo += "_evalMode-"+config.DATABASE["eval_mode"]
        dataInfo += "_dataAug-"+str(config.DATABASE["data_aug"])
        dataInfo += "_filter-BQ"+str(config.DATABASE["base_quality_threshold"])+"-MAPQ-"+str(config.DATABASE["mapq_threshold"])

        annoElems = config.AnnoCNVFile[args.dataSelect].split("/")
        dataInfo += "_AnnoFile-"+annoElems[-2]+":" +annoElems[-1]

        dataPath = dataPath + dataInfo

        if(config.DATABASE["eval_mode"]=="cross"):
            dataPath += "_testCrossSample-"+ args.dataSelect2

        if not os.path.exists(dataPath):
            cache_trainData(goldFile, bamFilePath, dataPath, args.dataSplit)
 
        ## model parameter is associated with the training data [depth bin size, annotation] 
        modelParamPath = "../experiment/model_param/"  
        dataInfo = args.dataSelect +"_"+config.DATABASE["count_type"] +"_bin"+str(binSize)+"_TRAIN"
        dataInfo += "_extendContext-" + str(config.DATABASE["extend_context"])
        dataInfo += "_dataAug-"+str(config.DATABASE["data_aug"])
        dataInfo += "_filter-BQ"+str(config.DATABASE["base_quality_threshold"])+"-MAPQ-"+str(config.DATABASE["mapq_threshold"])
        annoElems = config.AnnoCNVFile[args.dataSelect].split("/")
        dataInfo += "_AnnoFile-"+annoElems[-2]+":" +annoElems[-1]
        modelParamPath = modelParamPath + dataInfo

        ## training model
        globals()[args.model](dataPath, bk_dataPath, args.modelWeight, dataInfo)
    
