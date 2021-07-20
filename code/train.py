"""
Date: 2018-08-16, 2021-07-19
Author: Yao-zhong Zhang
Description:
This is the training function of UNet, CNN of RDBKE, which is used for breakpoint resolution
enhancement for read-depth based SV callers.
"""

# -*- coding: utf-8 -*-
from __future__ import division
from util import *

from keras import Input, models, layers, regularizers, metrics
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

# revision the model part
from models.model_unet import *
from models.model_multi_info import *

from models.model_baseline import *
from models.model_hyperOpt import *

from dataProcess.data_cacheLoading import *
from datetime import date

EPOCH=100

# model training settings
CB = [ callbacks.EarlyStopping(monitor="val_loss", patience=10, mode = "auto", restore_best_weights=True) ] 
VB = 0

TRAIL=100

############################################################
# Cross valdiation evluation
## not revised yet , check before using !
def train_CV(modelName, dataPath, bk_dataPath, modelParamPath, modelSavePath, dataInfo, loss, plotResult= False, plotTrainCurve=False):

        # all data loading, the data is pre-splited in the caching part.
        print("@ Loading data ...")
        x_data, y_data, rgs_data, bps_data, x_cnv, y_cnv, rgs_cnv, bps_cnv = loadData(dataPath, bk_dataPath, \
            prob_add=config.DATABASE["USEPROB"], seq_add=config.DATABASE["USESEQ"], gc_norm=config.DATABASE["GC_NORM"])
        
        # concatenate both train and test split for cross-validation
        x_data = np.concatenate((x_data, x_cnv), 0)
        y_data = np.concatenate((y_data, y_cnv), 0)
        bps_data = np.concatenate((bps_data, bps_cnv), 0)
        rgs_data = np.concatenate((rgs_data, rgs_cnv), 0)

        print("* After concatenating of all the data...")
        print(x_data.shape)
    
        print("- Loading data okay and start processing ...")
        y_data = y_data.reshape(y_data.shape[0], y_data.shape[1], 1)
        y_data_label = np.apply_along_axis(checkLabel, 1, y_data)
        print("BK=%d / Total=%d" %(np.sum(y_data_label), y_data_label.shape[0]))
 
        # using provided hyper-parameters
        if config.DATABASE["model_param"] != "":
            params = load_modelParam(config.DATABASE["model_param"])
            print("Loaded model parameters ...")
            print(params) 
        else:
            print("[Error!] Model's parameter file is not provided, please check!")
            exit(-1)   

        maxpooling_len = params["maxpooling_len"]
        conv_window_len = params["conv_window_len"]
        lr = params["lr"]
        batchSize = params["batchSize"]
        dropoutRate = params["DropoutRate"]

        # generate the model names according to model parameters
        modelInfo = "UNet-all_maxpoolingLen_"+"-".join([str(l) for l in maxpooling_len]) 
        modelInfo += "-convWindowLen_"+ str(conv_window_len)
        modelInfo += "-lr_"+ str(lr)
        modelInfo += "-batchSize" + str(batchSize)
        modelInfo += "-epoch"+str(params["epoch"])
        modelInfo += "-dropout"+str(dropoutRate)
        modelInfo += "-loss_" + str(loss)

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

            # small CV training option
            if config.DATABASE["small_cv_train"] == "small":
                print("+ Small train CV actived!!!! ")
                train_idx, test_idx = test_idx, train_idx

            signals = Input(name='input', shape=(x_data.shape[1], x_data.shape[2]), dtype=np.float32)

            model = globals()[modelName](signals, params["kernel_size"], params["conv_window_len"],  params["maxpooling_len"],  \
            params["stride"], True, params["DropoutRate"])

            if loss == "dice_loss":
                model.compile(optimizer=Adam(lr = lr) , loss= dice_loss, metrics=[dice_coef, iou_score])
            if loss == "bc_dice_loss":
                model.compile(optimizer=Adam(lr = lr) , loss= bc_dice_loss, metrics=[dice_coef, metrics.binary_accuracy])  
            if loss == "abc_dice_loss":
                model.compile(optimizer=Adam(lr = lr) , loss= abc_dice_loss, metrics=[dice_coef, metrics.binary_accuracy])
            if loss == "binary_crossentropy":
                model.compile(optimizer=Adam(lr = lr) , loss= "binary_crossentropy", metrics=[dice_coef, metrics.binary_accuracy])  
            if loss == "iou_loss":
                model.compile(optimizer=Adam(lr = lr) , loss= iou_loss, metrics=[dice_coef, iou_score, metrics.binary_accuracy])  
    
            if loss == "bk_loss":
                # packing y_data
                print("@ adding the the BK weighting information in the target label data...")
                w_matrix = genWeightMatrix(bps_data)
                w_matrix = w_matrix.reshape(w_matrix.shape[0], w_matrix.shape[1], 1)
                y_data_pack = np.concatenate((y_data, w_matrix), -1)
                model.compile(optimizer=Adam(lr = lr) , loss= bk_loss, metrics=[unpack_dice_score, unpack_binary_accuracy, unpack_iou_score])  

            ## model fitting, special processing for the self defined ones. 
            if loss == "bk_loss":
                history = model.fit(x_data[train_idx], y_data_pack[train_idx], epochs=params["epoch"], batch_size= batchSize, verbose=0, \
                        validation_split=0.2, callbacks=CB)
            else:
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

            figureSavePath= "../experiment/result/" + date.today().strftime("%Y%m%d") + "_"
            evluation_breakpoint(x_data[test_idx], rgs_data[test_idx], gold, pred, figureSavePath, False)

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

        
# non-cross validation setting for the u-net, first tailing
def train(modelName, dataPath, bk_dataPath, modelParamPath, modelSavePath, dataInfo, loss, plotTrainCurve=False, plotResult=False):

        # Data loading start
        print("@ Loading training data ...")
        x_data, y_data, rgs_data, bps_data, x_cnv, y_cnv, rgs_cnv, bps_cnv = loadData(dataPath, bk_dataPath, \
            prob_add=config.DATABASE["USEPROB"], seq_add=config.DATABASE["USESEQ"], gc_norm=config.DATABASE["GC_NORM"])

        print(x_data.shape)
        if("UNet" in modelName):
            y_data = y_data.reshape(y_data.shape[0], y_data.shape[1], 1)
        print(y_data.shape)
        
        y_data_label = np.apply_along_axis(checkLabel, 1, y_data)
        print("@ BK=%d / Total=%d" %(np.sum(y_data_label), y_data_label.shape[0]))       

        # model parameters loading
        if config.DATABASE["model_param"] != "":
            params = load_modelParam(config.DATABASE["model_param"])
        else:
            model_param_file  =  modelParamPath + ".TRAIL-"+ str(TRAIL)+ ".UNet.model.parameters.json"
            # check whether the model is hyperOpt tuned
            if not os.path.exists(model_param_file):
                tmpData = (x_data, y_data, y_data_label)
                do_hyperOpt(modelName, tmpData, TRAIL, model_param_file)
            params = load_modelParam(model_param_file)

        print("@ Model parameters are :")
        print(params)
        print("*"*40)

        maxpooling_len = params["maxpooling_len"]
        conv_window_len = params["conv_window_len"]
        lr = params["lr"]
        batchSize = params["batchSize"]
        dropoutRate = params["DropoutRate"]

        modelInfo =  modelName + "_maxpoolingLen_"+"-".join([str(l) for l in maxpooling_len]) 
        modelInfo += "-kernel_" + "-".join([str(l) for l in params["kernel_size"] ])
        modelInfo += "-convWindowLen_"+ str(conv_window_len)
        modelInfo += "-lr_"+ str(lr)
        modelInfo += "-batchSize" + str(batchSize)
        modelInfo += "-epoch"+str(params["epoch"])
        modelInfo += "-dropout"+str(dropoutRate)
        modelInfo += "-loss"+str(loss)

        if config.DATABASE["USEPROB"]: modelInfo += "_USEPROB"
        if config.DATABASE["USESEQ"]: modelInfo += "_USESEQ"
        if config.DATABASE["GC_NORM"]: modelInfo += "_GC-NORM"

        #########################################
        modelSaveName = os.path.basename(modelParamPath) + "_" + config.DATABASE["model_data_tag"] #modelName  # + "_" +modelInfo

        signals = Input(name='input', shape=(x_data.shape[1], x_data.shape[2]), dtype=np.float32)

        # call the networks
        model = globals()[modelName](signals, params["kernel_size"], params["conv_window_len"],  params["maxpooling_len"],  \
            params["stride"], True, params["DropoutRate"])

        if loss == "dice_loss":
            model.compile(optimizer=Adam(lr = lr) , loss= dice_loss, metrics=[dice_coef, iou_score])
        if loss == "bc_dice_loss":
            model.compile(optimizer=Adam(lr = lr) , loss= bc_dice_loss, metrics=[dice_coef, metrics.binary_accuracy])  
        if loss == "abc_dice_loss":
            model.compile(optimizer=Adam(lr = lr) , loss= abc_dice_loss, metrics=[dice_coef, metrics.binary_accuracy])
        if loss == "binary_crossentropy":
            model.compile(optimizer=Adam(lr = lr) , loss= "binary_crossentropy", metrics=[dice_coef, metrics.binary_accuracy])  
        if loss == "iou_loss":
            model.compile(optimizer=Adam(lr = lr) , loss= iou_loss, metrics=[dice_coef, iou_score, metrics.binary_accuracy])  
        
        if loss == "bk_loss":
            # packing y_data
            print("@ adding the the BK weighting information in the target label data...")
            w_matrix = genWeightMatrix(bps_data)
            w_matrix = w_matrix.reshape(w_matrix.shape[0], w_matrix.shape[1], 1)
            y_data = np.concatenate((y_data, w_matrix), -1)
            model.compile(optimizer=Adam(lr = lr) , loss= bk_loss, metrics=[unpack_dice_score, unpack_binary_accuracy, unpack_iou_score])  

        history = model.fit(x_data, y_data, epochs=params["epoch"], batch_size=batchSize, verbose=VB,callbacks=CB, validation_split=0.2)
        
        print "@ Saving model ..."
        model.save(modelSavePath + "/model/" + modelSaveName + ".h5" )
        plotTrainCurve = False

        model.summary()

        ###########################################################################################3
        # Results generation
        ###########################################################################################
        if plotTrainCurve:

            figureSavePath= modelSavePath + "/model_curve/" + modelSaveName + "_trainCurve.png"
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
        print("[*]: Train BreakPoints=%d / Total=%d" %(np.sum(y_data_label), y_data_label.shape[0]))

        ## evluation of the break point.
        figureSavePath= "../experiment/ROC/" + date.today().strftime("%Y%m%d") + "_"
        binary_eval(gold_cnv, pred_cnv, modelInfo, False, figureSavePath + "UNet" + ".png")
       
        figureSavePath= "../experiment/result/" + date.today().strftime("%Y%m%d") + "_"
        evluation_breakpoint(x_cnv, rgs_cnv, gold_cnv, pred_cnv, figureSavePath + "UNet_", False)
        print("\n")


if __name__ == "__main__":
    
        parser = argparse.ArgumentParser(description='Intra-bin segmentation for SV augmentation')

        parser.add_argument('--gpu', '-g', type=str, default="0", help='GPU ID for Training model.')
        parser.add_argument('--bin', '-b', type=int, default=1024, required=True, help='bin size of target region')
        parser.add_argument('--dataAug', '-da', type=int, default=0, help='Epochs of data augmentation')
        parser.add_argument('--model', '-m', type=str, default="UNet", required=True, help='UNet/CNN/SVM')
        parser.add_argument('--trainMode', '-tm', type=str, default="", help='Training model [, CV]')

        parser.add_argument('--ref_faFile', '-refa', type=str, default="", help='Reference files')      
        #parser.add_argument('--chr_prefix', '-chrPrefix', type=bool, default=True, help='Chromesome name Prefix status')   

        parser.add_argument('--dataSplit', '-ds', type=str, default="RandRgs",help='Data split setting')
        parser.add_argument('--testSplitPortion', '-tsp', type=float, default=0.2, help='Data split portion')

        parser.add_argument('--smallTrainCV', '-scv', type=str, default="normal", help='')
        
        parser.add_argument('--evalMode', '-em', type=str, default="single", required=True, help='Model evaluation mode')
        
        parser.add_argument('--dataSelect', '-d', type=str, default="na12878_7x", required=True, help='bam file')
        parser.add_argument('--dataSelect2', '-d2', type=str, default="", help='bam file for corss-sample testing')

        parser.add_argument('--modelParam', '-mp', type=str, default="", help='resign pre-determined model parameters')
        parser.add_argument('--loss', '-l', type=str, default="abc_dice_loss", help='losses of training the model')
        parser.add_argument('--bg_eval', '-be', type=bool, default=True, help='whether background region evaluated')

        # Loading target training regions
        parser.add_argument('--vcf', '-vcf', type=str, default="", required=True, help="vcf/bed annotation")
        parser.add_argument('--vcf2', '-vcf2', type=str, default="", help="second vcf/bed for testing")
        parser.add_argument('--vcf_ci', '-vcf_ci', type=int, default=500, help="Whether filtering PASS tags in the VCF")

        parser.add_argument('--vcf_filter', '-vcf_filter', type=bool, default=False, help="Whether filtering PASS tags in the VCF")
        parser.add_argument('--vcf_filter2', '-vcf_filter2', type=bool, default=False, help="Whether filtering PASS tags in the VCF")
        parser.add_argument('--fix_center', '-fix_center', type=bool, default=False, help="Fix the winodw and make the break point centering")
        parser.add_argument('--shift_low_bound', '-shift_low_bound', type=int, default=10, help="random shift low distance boundary to breakpoint data")

        # update pre-determined paramters from the command line
        args = parser.parse_args()
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        if args.ref_faFile != "" and args.ref_faFile != config.DATABASE["ref_faFile"]:
            config.DATABASE["ref_faFile"] = args.ref_faFile

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

        if args.vcf != "":
            config.DATABASE["vcf"] = args.vcf
            config.AnnoCNVFile[args.dataSelect] = args.vcf
        if args.vcf2 != "":
            config.DATABASE["vcf2"] = args.vcf2
            config.AnnoCNVFile[args.dataSelect2] = args.vcf2

        if args.vcf_ci != config.DATABASE["vcf_ci"]:
            config.DATABASE["vcf_ci"] = args.vcf_ci

        if args.vcf_filter != config.DATABASE["vcf_filter"] :
            config.DATABASE["vcf_filter"] = args.vcf_filter

        if args.vcf_filter2 != config.DATABASE["vcf_filter2"] :
            config.DATABASE["vcf_filter2"] = args.vcf_filter2

        if args.fix_center != config.DATABASE["fix_center"]:
            config.DATABASE["fix_center"] = args.fix_center

        if args.shift_low_bound != config.DATABASE["shift_low_bound"]:
            config.DATABASE["shift_low_bound"] = args.shift_low_bound

        if args.smallTrainCV != config.DATABASE["small_cv_train"]:
            config.DATABASE["small_cv_train"] = args.smallTrainCV

        binSize = config.DATABASE["binSize"]
        config.DATABASE["model_data_tag"] = args.model + "_" + args.dataSelect + "_b"+str(binSize) + "_tsp" + str(args.testSplitPortion)
        
        ANNOTAG=""

        ## background data first caching check
        ## background data is random sampling whole genome read depth data
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

        dataInfo += "_VCF-filter" if args.vcf_filter else ""
        dataInfo += "_BG-EVAL" if args.bg_eval else ""
        dataInfo += "_FixCenter" if args.fix_center else ""
        dataInfo += "_shiftLowBound_" + str(config.DATABASE["shift_low_bound"])
        dataInfo += "_vcf-ci_" + str(config.DATABASE["vcf_ci"])

        dataPath = dataPath + dataInfo

        if(config.DATABASE["eval_mode"]=="cross"):
            dataPath += "_testCrossSample-"+ args.dataSelect2

        # loading data from VCF, this is the key part, therefore except the name no affection for the current results!!!
        if config.DATABASE["vcf"] != "": 
            print("[*] Generating data in VCF mode ...")
            if config.DATABASE["vcf2"] != "": 
                cache_trainData_fromVCF([args.vcf, args.vcf2], bamFilePath, dataPath, args.testSplitPortion)
            else:
                cache_trainData_fromVCF([args.vcf], bamFilePath, dataPath, args.testSplitPortion)

        ## model parameter is associated with the training data [depth bin size, annotation] 
        modelParamPath = "../experiment/model_param/"  
        dataInfo = args.dataSelect +"_" + config.DATABASE["count_type"] + "_bin"+str(binSize) + "_TRAIN"
        dataInfo += "_extendContext-" + str(config.DATABASE["extend_context"])
        dataInfo += "_dataAug-"+str(config.DATABASE["data_aug"])
        dataInfo += "_filter-BQ"+str(config.DATABASE["base_quality_threshold"])+"-MAPQ-"+str(config.DATABASE["mapq_threshold"])
        annoElems = config.AnnoCNVFile[args.dataSelect].split("/")
        dataInfo += "_AnnoFile-"+annoElems[-2]+":" +annoElems[-1]
        modelParamPath = modelParamPath + dataInfo
        
        modelSavePath ="/home/yaozhong/working/1_Unet_IntraSV/experiment"

        # used for saving different train_tag_set.
        if args.dataSplit == "CV":
            if args.model == "SVM":
                SVM_CV(dataPath, bk_dataPath, modelParamPath, modelSavePath, dataInfo, plotResult= False, plotTrainCurve=False)
            elif args.model == "CNN":
                CNN_CV(dataPath, bk_dataPath, modelParamPath, modelSavePath, dataInfo, plotResult= False, plotTrainCurve=False)        
            else:  
                train_CV(args.model, dataPath, bk_dataPath, modelParamPath, modelSavePath, dataInfo, args.loss, plotTrainCurve=False, plotResult=False)
        else:
            if args.model == "SVM":
                SVM(dataPath, bk_dataPath, modelParamPath, modelSavePath, dataInfo)
            elif args.model == "CNN":
                CNN(dataPath, bk_dataPath, modelParamPath, modelSavePath, dataInfo, plotTrainCurve=False, plotResult=False)   
            else: 
                train(args.model, dataPath, bk_dataPath, modelParamPath, modelSavePath , dataInfo, args.loss, plotTrainCurve=False, plotResult=False)
