"""
Date: 2018-11-3
Description: Hyperopt for search the best model performance
"""

from model_baseline import *
from model_unet import UNet_networkstructure_basic
#from train import *
from util import *

from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials, space_eval
#from hyperopt.mongoexp import MongoTrials
import json


## the data should be expose to the level of objective function
#CB = [ callbacks.EarlyStopping(monitor="val_dice_coef", patience=10, restore_best_weights=True) ] 


# define parameter space
space ={
        'C':hp.uniform('C', 0, 2),
        'gamma': hp.loguniform('gamma', -8, 2),
        'kernel': hp.choice('kernel', ['rbf', 'poly', 'sigmoid'])
}

def objective(args):

    model = svm.SVC(**args)
    model.fit(x_data_opt, y_data_opt):
    
    t_cnv = model.predict(x_cnv_opt).ravel()

    return {'loss': , 'status':STATUS_OK}



# define objective function
def objective(params):

    #model define
    rd_input = Input(shape=(x_data_opt.shape[1], x_data_opt.shape[-1]), dtype='float32', name="rd")
    
    model = UNet_networkstructure_basic(rd_input, params["conv_window_len"],  \
            params["maxpooling_len"], True , params["DropoutRate"])
    
    model.compile(optimizer=Adam(lr = params["lr"]) , loss= dice_coef_loss, metrics=[dice_coef])
    model.fit(x_data_opt, y_data_opt, epochs=params["epoch"], batch_size=params["batchSize"], verbose=0)
    
    t_cnv = model.predict(x_cnv_opt, verbose=0)
    pred_cnv = (t_cnv > 0.5).astype(np.float32).reshape(t_cnv.shape[0], t_cnv.shape[1])
    gold_cnv = y_cnv_opt.astype(np.float32)
    
    #score 
    df_cnv = dice_score(gold_cnv, pred_cnv)

    return {'loss':df_cnv*(-1), 'status':STATUS_OK}


if __name__ == "__main__":
    
        parser = argparse.ArgumentParser(description='DL Based Break Point Detection')
        parser.add_argument('--gpu', '-g', type=str, default="3", help='Assign GPU for Training model.')
        parser.add_argument('--bin', '-b', type=int, default=1000, help='screening window length.')
        parser.add_argument('--dataAug', '-da', type=int, default=0, help='Number of additional proportional samples to gen.')
        parser.add_argument('--model', '-m', type=str, default="CNN", help='Model type for training break point.')
        
        parser.add_argument('--dataSplit', '-ds', type=str, default="CV", help='Model type for training break point.')
        parser.add_argument('--evalMode', '-em', type=str, default="single", help='Model type for training break point.')
        
        parser.add_argument('--dataSelect', '-d', type=str, default="na12878_7x", help='Training data sample.')
        parser.add_argument('--dataSelect2', '-d2', type=str, default="", help='cross sample evluation sample.')

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

        binSize = config.DATABASE["binSize"]

        ANNOTAG=""
        ## background ata first caching check
        bk_dataPath = "../data/data_cache/"+ ANNOTAG + args.dataSelect \
                +"_"+config.DATABASE["count_type"] \
                +"_bin"+str(binSize)+"_GENOMESTAT_"
        
        # genome information 
        bk_dataPath += "SampleRate-" + str(config.DATABASE["genomeSampleRate"]) 
        bk_dataPath += "_Filter-Mappability-"+ str(config.DATABASE["mappability_threshold"])
        
        bamFilePath = config.BAMFILE[args.dataSelect]

        if not os.path.exists(bk_dataPath):
            cache_genome_statistics(bamFilePath, bk_dataPath, config.DATABASE["genomeSampleRate"])
        
        bk_dataPath = [bk_dataPath]
        ## training data first cachcing check
        if config.DATABASE["eval_mode"] == "cross":
            assert(args.dataSelect2 != "")
            goldFile = [config.AnnoCNVFile[args.dataSelect], config.AnnoCNVFile[args.dataSelect2]]
            bamFilePath = [config.BAMFILE[args.dataSelect], config.BAMFILE[args.dataSelect2]]
        else:
            assert(args.dataSelect2 == "")
            goldFile = [config.AnnoCNVFile[args.dataSelect]]
            bamFilePath = [config.BAMFILE[args.dataSelect]]


        dataPath = "../data/data_cache/"+ ANNOTAG
        
        dataInfo = args.dataSelect +"_"+config.DATABASE["count_type"] +"_bin"+str(binSize)+"_TRAIN"
        dataInfo += "_extendContext-" + str(config.DATABASE["extend_context"])
        dataInfo += "_dataSplit-" + config.DATABASE["data_split"] # dataSplit information 
        dataInfo += "_evalMode-"+config.DATABASE["eval_mode"]
        dataInfo += "_dataAug-"+str(config.DATABASE["data_aug"])
        dataInfo += "_filter-BQ"+str(config.DATABASE["base_quality_threshold"])+"-MAPQ-"+str(config.DATABASE["mapq_threshold"])
    
        annoElems = config.AnnoCNVFile[args.dataSelect].split("/")
        dataInfo += "_AnnoFile-"+annoElems[-2]

        dataPath = dataPath + dataInfo

        if(config.DATABASE["eval_mode"]=="cross"):
            dataPath += "_testCrossSample-"+ args.dataSelect2
            
        if not os.path.exists(dataPath):
            cache_trainData(goldFile, bamFilePath, dataPath, args.dataSplit)

        # loading data function
        x_data, y_data, rgs_data, x_cnv, y_cnv, rgs_cnv = loadData(dataPath, bk_dataPath, prob_add=USEPROB, seq_add=USESEQ)

        y_data = y_data.reshape(y_data.shape[0], y_data.shape[1], 1)
        y_data_label = np.apply_along_axis(checkLabel, 1, y_data)

        trials = Trials()
        best = fmin(objective, space, algo=tpe.suggest, max_evals=100, trials=trials, verbose=1)
        print best
        print (trials.best_trial)


#######################################################
# call API 
#######################################################
def do_hyperOpt_SVM(data, tryTime = 10, paramFile=None):
    
    x_data_tmp, y_data_tmp, y_data_label = data
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.DATABASE["rand_seed"])
    train_idx, dev_idx =  next(kfold.split(x_data_tmp, y_data_label))

    global x_data_opt, y_data_opt, x_cnv_opt, y_cnv_opt
    x_data_opt = x_data_tmp[train_idx]
    y_data_opt = y_data_tmp[train_idx]
    x_cnv_opt  = x_data_tmp[dev_idx]
    y_cnv_opt  = y_data_tmp[dev_idx]

    print "* Model hyper parmaters tunning start ..."
    
    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=tryTime, trials=trials, verbose=1)

    param_dic = space_eval(space, best)
    
    jd = json.dumps(param_dic)
    output = open(paramFile, "w")
    output.write(jd)
    output.close()



def load_modelParam(paramFile):
    
    with open(paramFile, "r") as f:
        param_dic = json.load(f)
        return param_dic

    
"""

space ={
        'conv_window_len':hp.choice('conv_window_len',[3,5,7,9]),
        'maxpooling_len':hp.choice('maxpooling_len', [[2,2,2,2,2,2], [10,5,2,2,5,10], [5,5,4,4,5,5],[8,5,5,5,5,8], [10,10,2,2,10,10]]),
         # 256 will be out of memory
        'batchSize': hp.choice('batchSize', [8,16,32,64,128]),
        'lr': hp.choice('lr', [ 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1]),
        'DropoutRate': hp.choice("DropoutRate", [0.5, 0.3, 0.2, 0]),
        'BN': hp.choice("BN", [True, False]),
        'epoch': hp.choice("epoch",[10])
}

"""
