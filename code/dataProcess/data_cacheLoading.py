"""
Date: 2018-11-05 
Description: cache and loading basic bin-based signals: Depth-of-coverage, Sequence, bin region information. 

"""
from data_genBinData import*
from data_genomeStats import *

from keras.utils import to_categorical


## y_data calcuate the label type
def checkLabel(x):
    total = np.sum(x)
    if total == 0:
        return 0
    else:
        return 1

def printBasicStat(x_data, y_data):
    
    y_label = np.apply_along_axis(checkLabel, 1, y_data)
    idx_bg = [ i for i in range(len(y_label)) if y_label[i] == 0 ]
    idx_bk = [ i for i in range(len(y_label)) if y_label[i] == 1 ]
    
    logger.info("\n[BIN_DATA]:Sample label number Positive=%d, Negative=%d" %(len(idx_bk), len(idx_bg)))
    
    print("[BIN_DATA]:Background data [m=%f, std=%f]" %(np.mean(x_data[idx_bg]), np.std(x_data[idx_bg])))
    print("[BIN_DATA]:Break point containing [m=%f, std=%f]" %(np.mean(x_data[idx_bk]), np.std(x_data[idx_bk])))

# 2018-12-28 GC normalization, based on the pre-calcuated gc caching data
def GC_count_dic_gen(gc_mrd_table):

    gc_count_dic = {}

    for i in range(10):
        gc_count_dic[i] = []

    for i in gc_mrd_table.keys():
        rg = int(i/10)
        gc_count_dic[rg].append(gc_mrd_table[i])

    for i in gc_count_dic.keys():
        if len(gc_count_dic[i]) == 0:
            gc_count_dic[i] = 0
        else:
            gc_count_dic[i] = np.mean(gc_count_dic[i])

    return gc_count_dic

def plot_GC_count_figure(rd_data, gc_data, figName):

    rd_x = [np.mean(rd_data[i]) for i in range(len(rd_data))]
    fig = plt.figure()
    plt.scatter(gc_data, rd_x, color="blue")
    plt.grid(True)
    plt.savefig(figName)
    plt.close("all")


def GC_normalization(m_rd, gc_count_dic, rd_data, gc_data):
    
    # generate the plot, before and after normalization
    plot_GC_count_figure(rd_data, gc_data, "../experiment/beforeGCNorm.png")

    for i in range(len(rd_data)):
        
        gc_key = int(gc_data[i]*10)
        factor = (m_rd+1)/(gc_count_dic[gc_key]+1)
        rd_data[i] = rd_data[i] * factor
    
    #print np.min(gc_data), np.max(gc_data)
    

    plot_GC_count_figure(rd_data, gc_data, "../experiment/AfterGCNorm.png")

    return rd_data



# dataPath and bk_dataPath is generated based on the options
## this is for inner sample evluation case. 
def loadData(dataPath, bk_dataPath, prob_add=False, seq_add=False, gc_norm=False):

        # loading background genome sampling
        m_rd, std_rd, md_rd, gc_mrd_table = load_genome_statistics(bk_dataPath[0])

        visal_rd_genome(bk_dataPath[0],False)
        
        if config.DATABASE["eval_mode"] == "cross":
            m_rd2, std_rd2, md_rd2, gc_mrd_table2 = load_genome_statistics(bk_dataPath[1])
            visal_rd_genome(bk_dataPath[1],False)
        
        print("-- loading cache genome information done!")

        # loading CNV training samples
        x_train, y_train, seq_train, rgs_train,  gc_train, \
        x_test, y_test, seq_test, rgs_test, gc_test   = load_cache_trainData(dataPath)
        
        printBasicStat(x_train, y_train)
        print("-- loading cache bin data information done!")

        ## CNV data set filtering
        indicator = np.apply_along_axis(check_all_zero, 1, x_train)
        select, un_select = [], []
        
        for i, tag in enumerate(indicator):
            if tag == False:
                select.append(i)
            else:
                un_select.append(i)
        
        print("\t * [Filtering]:Ignoring All-ZERO input in the Training dataset, Number is [%d]" %(len(un_select)))
        x_train = x_train[select]
        y_train = y_train[select]
        seq_train = seq_train[select]
        rgs_train = rgs_train[select]
        gc_train = gc_train[select]

        indicator = np.apply_along_axis(check_all_zero, 1, x_test)
        select, un_select = [], []
        for i, tag in enumerate(indicator):
            if tag == False:
                select.append(i)
            else:
                un_select.append(i)
        
        print("\t * [Filtering]:Ignoring ALL-ZERO input in the Testing set, Number is %d" %(len(un_select)))
        x_test = x_test[select]
        y_test = y_test[select]
        seq_test = seq_test[select]
        rgs_test = rgs_test[select]
        gc_test = gc_test[select]

        logger.debug("\n * [Check point] all the test should be the same for each run")
        logger.debug(rgs_test[1:6])
        
        ###########################################################################################

        if prob_add :
            """
            r, p = get_NegBinomial_params(bk_dataPath)
            x_train_prob = ss.nbinom.pmf(x_train, r, p)
            x_test_prob =  ss.nbinom.pmf(x_test, r, p)
            """
            if config.DATABASE["eval_mode"] != "cross":
                x_train_prob = ss.norm.pdf(x_train, m_rd, std_rd)
                x_test_prob = ss.norm.pdf(x_test, m_rd, std_rd)
            else:
                x_train_prob = ss.norm.pdf(x_train, m_rd, std_rd)
                x_test_prob = ss.norm.pdf(x_test, m_rd2, std_rd2)


        # Standarization
        if config.DATABASE["eval_mode"] != "cross":
            x_train = (x_train - m_rd)/std_rd
            x_test = (x_test - m_rd)/std_rd
        else:
            x_train = (x_train - m_rd)/std_rd
            x_test = (x_test - m_rd2)/std_rd2
    

        # transform the data to tensro
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)


        if prob_add:
            
            x_train_prob = x_train_prob.reshape(x_train_prob.shape[0], x_train_prob.shape[1], 1)
            x_test_prob = x_test_prob.reshape(x_test_prob.shape[0], x_test_prob.shape[1], 1)

            ## combine
            # x_train = np.concatenate((x_train_prob, x_train), -1)
            # x_test = np.concatenate((x_test_prob, x_test), -1)
            
            ## replace
            x_train = x_train_prob
            x_test =  x_test_prob
        

        # concatenate the data with sequence information, not validated!
        if seq_add:
    
            seq_train = to_categorical(seq_train)
            seq_test = to_categorical(seq_test)
            
            # combine into kernals
            x_train = np.concatenate((x_train, seq_train), -1)
            x_test = np.concatenate((x_test, seq_test), -1)


        if gc_norm:

            gc_count_dic = GC_count_dic_gen(gc_mrd_table)
            x_train = GC_normalization(md_rd, gc_count_dic, x_train, gc_train)

            if config.DATABASE["eval_mode"] == "cross":
                gc_count_dic2 = GC_count_dic_gen(gc_mrd_table2)
                x_test =  GC_normalization(md_rd2, gc_count_dic2, x_test,  gc_test)
            else:
                x_test =  GC_normalization(md_rd, gc_count_dic,  x_test,  gc_test)


        return (x_train, y_train, rgs_train,  x_test, y_test, rgs_test)


