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
            filePath = figurePathName + "/bg-"+"_".join(rgs[i])
            figName = "Background bin,"
        else:
            filePath = figurePathName + "/breakPoint-"+"_".join(rgs[i])
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

        plt.yticks((4,3,2,1,0,-1,-2,-3,-4, -5,-7), (4,3,2,1,0,-1,-2,-3,-4, "Gold", "Pred"))
        plt.title(figName)
        plt.savefig(filePath)
        plt.close("all")
    
    print "* The segment files are saved in ", figurePathName

"""
#########################################
#  Types of data visualization
#########################################
def getPCAcomponent(data, compNum):

    pca = PCA(n_components=compNum)
    pca.fit(data)
    X = pca.transform(data)
    
    #print(pca.explained_variance_)
    print(X.shape)

    return X



#########################################
# Input visualization
########################################

def loadData(dataPath, bk_dataPath, dataDN, seq_add=False):

        m_rd, std_rd, md_rd, gc_mrd_table = load_genome_statistics(bk_dataPath)
        logger.info("** Genome statistics: mean=%f, std=%f, median=%s, gcNum=%d **" %(m_rd, std_rd, md_rd, len(gc_mrd_table)))

        x_train, y_train, seq_train, x_test, y_test, seq_test, rgs_test = load_cache_data(dataPath)
        
        r, p = NB_GC_Fit(bk_dataPath)
        x_train_prob = ss.nbinom.pmf(x_train, r, p)
        x_test_prob =  ss.nbinom.pmf(x_test, r, p)

        # Normalization, mainly focused on the training data analysis
        x_train_norm = (x_train - m_rd)/std_rd
        x_test_norm = (x_test - m_rd)/std_rd

        ## 1. investigate the distribution difference between normal and bk-containing issues.
        num_sample = x_train.shape[0]
        nr, bk, nr_prob, bk_prob = [], [], [], []
        nr_idx, bk_idx = [], []
    
        for i in range(num_sample):
            if y_train[i] != 0:
                bk.extend(x_train_norm[i])
                bk_prob.extend(x_train_prob[i])
                bk_idx.append(i)
            else:
                nr.extend(x_train_norm[i])
                nr_prob.extend(x_train_prob[i])
                nr_idx.append(i)

        print("** [Break Point]-sample:")
        print(">> RD-mean=%f, RD-var=%f" %(np.mean(bk), np.std(bk)))
        print(">> RD_prob-mean=%f, RD_prob-var=%f\n" %(np.mean(bk_prob), np.std(bk_prob)))

        print("** Norm-sample:")
        print(">> RD-mean=%f, RD-var=%f" %(np.mean(nr), np.std(nr)))
        print(">> RD_prob-mean=%f, RD_prob-var=%f\n" %(np.mean(nr_prob), np.std(nr_prob)))

        
        ### plot hist
    
        fig = plt.figure()
        plt.hist(bk, color="blue", bins=100)
        plt.hist(nr, color="red", bins=100)
        figName = "../figures/"+ dataDN + "/train_rd_hist.png"
        plt.savefig(figName)
        plt.close("all")


        fig = plt.figure()
        plt.hist(bk, color="blue", bins=100)
        plt.hist(nr, color="red", bins=100)
        figName = "../figures/"+ dataDN + "/train_rd-prob_hist.png"
        plt.savefig(figName)
        plt.close("all")

        
        ## 2. quantify the difference between RD and Prob(RD).
        
        random.seed(config.DATABASE["rand_seed"])
	random.shuffle(nr_idx)
        random.shuffle(bk_idx)

        for i in range(10):
            idx = nr_idx[i]
            rd = x_train_norm[i]
            rd_prob = x_train_prob[i]


            fig = plt.figure()
            plt.plot(range(len(rd)), rd , color="blue")
            plt.plot(range(len(rd_prob)), rd_prob, color="red") 
            figName = "../figures/"+ dataDN + "/train_rd-prob_plot_"+ str(i)  +".png"
            plt.savefig(figName)
            plt.close("all")


            fig = plt.figure()
            plt.plot(range(len(rd)), rd , color="blue")
            figName = "../figures/"+ dataDN + "/train_rd-prob_plot_"+ str(i)  +"-rd.png"
            plt.savefig(figName)
            plt.close("all")


            fig = plt.figure()
            plt.plot(range(len(rd_prob)), rd_prob, color="red") 
            figName = "../figures/"+ dataDN + "/train_rd-prob_plot_"+ str(i)  +"-prob.png"
            plt.savefig(figName)
            plt.close("all")


        

        ### random select positions and plot


        ## 3. GC-content normlization and compared before and after.




 
if __name__ == "__main__":
    
        parser = argparse.ArgumentParser(description='DL Based Break Point Detection')
        parser.add_argument('--gpu', '-g', type=str, default="3", help='Assign GPU for Training model.')
        parser.add_argument('--bin', '-b', type=int, default=1000, help='screening window length.')
        parser.add_argument('--dataAug', '-da', type=int, default=0, help='Number of additional proportional samples to gen.')
        parser.add_argument('--model', '-m', type=str, default="CNN", help='Model type for training break point.')
        parser.add_argument('--dataSplit', '-s', type=float, default=0.2, help='Break point data split proportion for training and testing.')
        parser.add_argument('--dataSelect', '-d', type=str, default="300x", help='Selected different depth of NA12878 data.')
        
        args = parser.parse_args()

	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        if args.bin != config.DATABASE["binSize"]:
            config.DATABASE["binSize"] = args.bin
            logger.info("!!! Updating the bin size to %d" %(config.DATABASE["binSize"]))

        if args.dataAug != config.DATABASE["data_aug"]:
            config.DATABASE["data_aug"] = args.dataAug
    
        binSize = config.DATABASE["binSize"]

	dataPath = "/data/workspace/exp/data_cache/multiY_bk_rd_b"+str(binSize)+"_"
	goldFile = "../data/NA12878_1000GP_gold.txt"
        bamDataSelect={"7x":"/home/dev/NA12878.mapped.ILLUMINA.bwa.CEU.low_coverage.20121211.rmdup.bam", \
                "60x":"/home/dev/NA12878.mapped.ILLUMINA.bwa.CEU.high_coverage_pcr_free.20130906.rmdup.bam", \
                "300x":"/data/Dataset/GIAB/JP201712_data/HG001_NA12878/HG001.hs37d5.300x.bam"}

        bamFilePath = bamDataSelect[args.dataSelect]
	dataPath = dataPath + os.path.basename(goldFile)+ "_" +os.path.basename(bamFilePath) + "_dataSplit-" + str(args.dataSplit)
	if config.DATABASE["data_aug"] > 0:
            dataPath += "_dataAug-"+str(config.DATABASE["data_aug"])+"X_randSeed-"+str(config.DATABASE["rand_seed"])
            
        if not os.path.exists(dataPath):
            cache_data(goldFile, bamFilePath, dataPath, args.dataSplit)
 
        # do the background data analysis, this will be only determined by the binSize and different file. 
        sampleRate = 0.01
        
	bk_dataPath = "/data/workspace/exp/data_cache/genome_b"+str(binSize)+"_" + os.path.basename(bamFilePath) + "_SampleRate-" + str(sampleRate)
        if not os.path.exists(bk_dataPath):
            cache_genome_statistics(bamFilePath, bk_dataPath, sampleRate)

        logger.info("*** Now training model with bin=%d, data=%s, dataAug=%d, model=%s, dataSplit=%s, GenomeSampleRate=%f *** " %(args.bin, args.dataSelect, args.dataAug, args.model, str(args.dataSplit), sampleRate))
        #globals()[args.model](dataPath, bk_dataPath, "../keras_model/bk_"+args.model+".h5", 50)

        loadData(dataPath, bk_dataPath, args.dataSelect)
"""
