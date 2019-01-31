"""
Date: 2018-08-16
Author: Yao-zhong Zhang
Description: the basic training for the keras model for the break point detection problem.

2018-09-19: revised the model for the multi-class classification and evluation metrics.
2018-10-09: Add the autoencoder for the classification task. [with/without attention]
"""

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

EPOCH=100

# model training settings
CB = [ callbacks.EarlyStopping(monitor="val_loss", patience=10, mode = "auto", restore_best_weights=True) ] 
VB = 0

USESEQ = False
USEPROB = False
GC_NORM = False

TRAIL=30

# revised 
def UNet_CV(dataPath, bk_dataPath, modelParamPath, modelSavePath, dataInfo, plotResult= False, plotTrainCurve=False):

        # all data loading
        x_data, y_data, rgs_data, x_cnv, y_cnv, rgs_cnv = loadData(dataPath, bk_dataPath, prob_add=USEPROB, seq_add=USESEQ, gc_norm=GC_NORM)
        
        y_data = y_data.reshape(y_data.shape[0], y_data.shape[1], 1)
        y_data_label = np.apply_along_axis(checkLabel, 1, y_data)
        print("BK=%d / Total=%d" %(np.sum(y_data_label), y_data_label.shape[0]))
 
        # model parameters loading
        if config.DATABASE["model_param"] != "":
            params = load_modelParam(config.DATABASE["model_param"])
        else:

            model_param_file  =  modelParamPath + ".TRAIL-"+ str(TRAIL)+ ".UNet.model.parameters.json"
            # check whether the model is hyperOpt tuned
            if not os.path.exists(model_param_file):
                tmpData = (x_data, y_data, y_data_label)
                do_hyperOpt(tmpData, TRAIL, model_param_file)

            params = load_modelParam(model_param_file)
        
        print params   

        maxpooling_len = params["maxpooling_len"]
        conv_window_len = params["conv_window_len"]
        lr = params["lr"]
        batchSize = params["batchSize"]
        dropoutRate = params["DropoutRate"]

        modelInfo = "UNet-all_maxpoolingLen_"+"-".join([str(l) for l in maxpooling_len]) 
        modelInfo += "-convWindowLen_"+ str(conv_window_len)
        modelInfo += "-lr_"+ str(lr)
        modelInfo += "-batchSize" + str(batchSize)
        modelInfo += "-epoch"+str(params["epoch"])
        modelInfo += "-dropout"+str(dropoutRate)

        if USEPROB: modelInfo += "_USEPROB"
        if USESEQ: modelInfo += "_USESEQ"
        if GC_NORM: modelInfo += "_GC-NORM"

        ## K-fold cross validation
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.DATABASE["rand_seed"])
        cv_scores, cv_bk_scores, cv_bg_scores, cv_auc, cnv_scores= [], [], [], [], []
        cv_sensitivity, cv_FDR = [], []

        index = 0
        for train_idx, test_idx in kfold.split(x_data, y_data_label): 

            index = index + 1

            rd_input = Input(shape=(x_data.shape[1], x_data.shape[-1]), dtype='float32', name="rd")
            #model = UNet_networkstructure_basic(rd_input, conv_window_len, maxpooling_len)
            model = UNet_networkstructure_basic(rd_input, conv_window_len, maxpooling_len, True, dropoutRate)
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

        # for the convolutional output
        y_data = y_data.reshape(y_data.shape[0], y_data.shape[1], 1)
        y_data_label = np.apply_along_axis(checkLabel, 1, y_data)
        print("BK=%d / Total=%d" %(np.sum(y_data_label), y_data_label.shape[0]))       

        # model parameters loading
        if config.DATABASE["model_param"] != "":
            params = load_modelParam(config.DATABASE["model_param"])
        else:

            model_param_file  =  modelParamPath + ".TRAIL-"+ str(TRAIL)+ ".UNet.model.parameters.json"
        
            # check whether the model is hyperOpt tuned
            if not os.path.exists(model_param_file):
                tmpData = (x_data, y_data, y_data_label)
                do_hyperOpt(tmpData, TRAIL, model_param_file)
        
            params = load_modelParam(model_param_file)

        print "*"*40
        print params
        print "*"*40

        maxpooling_len = params["maxpooling_len"]
        conv_window_len = params["conv_window_len"]
        lr = params["lr"]
        batchSize = params["batchSize"]
        dropoutRate = params["DropoutRate"]

        modelInfo = "UNet_maxpoolingLen_"+"-".join([str(l) for l in maxpooling_len]) 
        modelInfo += "-convWindowLen_"+ str(conv_window_len)
        modelInfo += "-lr_"+ str(lr)
        modelInfo += "-batchSize" + str(batchSize)
        modelInfo += "-epoch"+str(params["epoch"])
        modelInfo += "-dropout"+str(dropoutRate)

        if USEPROB: modelInfo += "_USEPROB"
        if USESEQ: modelInfo += "_USESEQ"
        if GC_NORM: modelInfo += "_GC-NORM"

        #########################################
        modelSaveName = os.path.basename(modelParamPath) + "|" +modelInfo

        rd_input = Input(shape=(x_data.shape[1], x_data.shape[-1]), dtype='float32', name="rd")
        model = UNet_networkstructure_basic(rd_input, conv_window_len, maxpooling_len, True, dropoutRate)

        model.compile(optimizer=Adam(lr = lr) , loss= dice_coef_loss, metrics=[dice_coef])
        history = model.fit(x_data, y_data, epochs=params["epoch"], batch_size=batchSize, verbose=VB \
        ,callbacks=CB, validation_split=0.2)

        #model.summary()
        
        print "@ Saving model ..."
        model.save(modelSavePath +"/"+ modelSaveName +".h5" )


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


        print "\n=========== [DATA/MODEL information] ============="
        print "[" + dataInfo+ "]"
        print("Train BK=%d / Total=%d" %(np.sum(y_data_label), y_data_label.shape[0]))

        binary_eval(gold_cnv, pred_cnv, modelInfo)



if __name__ == "__main__":
    
        parser = argparse.ArgumentParser(description='Intra-bin SV segmentation for base-pair read-depth signals')
        parser.add_argument('--gpu', '-g', type=str, default="3", help='GPU ID for Training model.')
        parser.add_argument('--bin', '-b', type=int, default=1000, required=True, help='bin size of target region')
        parser.add_argument('--dataAug', '-da', type=int, default=0, help='Epochs of data augmentation')
        parser.add_argument('--model', '-m', type=str, default="UNet", required=True, help='Model names UNet/CNN/SVM')
        
        parser.add_argument('--dataSplit', '-ds', type=str, default="RandRgs",help='Data split setting')
        parser.add_argument('--evalMode', '-em', type=str, default="single", required=True, help='Model evaluation mode')
        
        parser.add_argument('--dataSelect', '-d', type=str, default="na12878_7x", required=True, help='bam file')
        parser.add_argument('--dataSelect2', '-d2', type=str, default="", help='bam file for corss-sample testing')

        parser.add_argument('--modelParam', '-mp', type=str, default="", help='resign pre-determined model parameters')

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

        ## background data first caching check
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
        globals()[args.model](dataPath, bk_dataPath, modelParamPath, "../experiment/model/", dataInfo)


