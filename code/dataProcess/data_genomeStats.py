"""
Data:	2018-08-14
Author: Yao-zhong Zhang @ IMSUT

Description:
Segment the whole genome into non-overlapping equal bin size. 
and Calculate the statisitcs information for the genome.

Cached data:
    1. RD (mean, std), which will be used for the data normalization
    2. GC and RD correlation table, which will be used for the GC content bias corretion.
    3. Parameters of RD to the Negative Binomial data distribtion, for potential Input data augmentation.
"""

from util import *
from data_bam2rd import *
from data_GC import *
from data_Mappability import *

import scipy.stats as ss
import scipy.optimize as so
#from fit_nbinom import *
import statsmodels.api as sm

"""
Return: m(rd), rd-gc-vec
based on the GC value, plot RD and GC corrleation.
Random sampling whole genome and calcuate the read-depth distriubtion. 
"""

def getSampleBG(bamFile, sampleFold=0.1):

    logger.info("\t================ Calcuate the Genomic Statistics based on bam File =================")
    random.seed(config.DATABASE["rand_seed"])

    # chrome scale regions
    rgs = get_chr_region(config.DATABASE["chrLen_info"])

    maVec, gcVec, rgVec = [],[],[]
    binSize = config.DATABASE["binSize"]

    for rg in rgs:
        
        if config.DATABASE["chr_prefix"]:
            rg = normChrName(rg, 1)
        else:
            rg = normChrName(rg, 0)

        chr_rgVec = getRegion_split(rg, binSize)
        rgVec.extend(chr_rgVec)

        chr_maVec = getChr_mappability_parallel_pool(normChrName(rg, 1), binSize, config.DATABASE["mappability_file"])
	maVec.extend(np.nan_to_num(chr_maVec))

    # Do mappability filter first, note if no mappability is needed just set the threshold to 0. 	
    idx = [ i for i,ma in enumerate(maVec) if ma > config.DATABASE["mappability_threshold"] ]
    logger.debug(">> Filtering the low mappability sample unit from [%d] to [%d] " %(len(maVec), len(idx)))
    maVec = np.array(maVec, dtype=np.float32)[idx]
    rgs = [ rgVec[i] for i in idx ]
	
    # random shuffling and get the fold proportion part.
    idx = range(len(maVec))	
    random.shuffle(idx)

    idx = idx[ 0:int(len(maVec)*sampleFold) ]
    maVec = maVec[idx]
    rgs = [rgs[i] for i in idx]
	
    #rdVec = get_RC_parallel_nobin(bamFile, rgs)
    rdVec = get_RDVec_parallel_nobin(bamFile, rgs)

    gcVec= getRegions_GC_parallel_nobin(config.DATABASE["ref_faFile"], rgs)
    gcVec = [ int(round(x*100)) for x in gcVec]
    
    # calcuating the basic statistics and do normalization according to the gc
    m_rd, std_rd, md_rd = np.mean(rdVec), np.std(rdVec), np.median(rdVec)
    print("** The basic statistics for the genome is [mead=%f, std=%f, median=%f]" %(m_rd, std_rd, md_rd))
    rd_basic =[m_rd, std_rd, md_rd]

    #2. Calcuate the potential GC-RD table.
    gc_rd_dic , gc_rd_table = {}, {}
    for i in range(len(gcVec)):
        gc = gcVec[i]
        if gc not in gc_rd_dic.keys():
            gc_rd_dic[gc] = []
        gc_rd_dic[gc].append(np.mean(rdVec[i,:]))
    
    gc_list, mrd_list=[],[]
    for gc in gc_rd_dic.keys():
        gc_list.append(gc)
        mrd_list.append(np.median(gc_rd_dic[gc]))

    return rd_basic, gc_list, mrd_list, rdVec


# Caching and loading genome background statistic data.
def cache_genome_statistics(bamFilePath, bk_dataPath, sampleRate):

    print("[Genome]: Caching %f percent of  whole genomes random sampling data... " %(sampleRate*100))
    rd_basic, gc_list, mrd_list, rdVec = getSampleBG(bamFilePath, sampleRate)

    with h5py.File(bk_dataPath, 'w') as hf:
 
        hf.create_dataset("rd_basic", data=rd_basic)
	hf.create_dataset("gc_list", data=gc_list)
        hf.create_dataset("mrd_list", data=mrd_list)
        hf.create_dataset("rdVec", data=rdVec)

def load_genome_statistics(bk_dataPath):

    with h5py.File(bk_dataPath, 'r') as hf:
        rd_basic = hf["rd_basic"][:]
        gc_list = hf["gc_list"][:]
        mrd_list = hf["mrd_list"][:]
        rdVec = hf["rdVec"][:]

        zipped = zip(gc_list, mrd_list)
        zipped.sort(key=lambda x:x[0])
        gc_mrd_table={}
        for gc, mrd in zipped:
            gc_mrd_table[gc] = mrd

        m_rd, std_rd, md_rd = rd_basic[0], rd_basic[1], rd_basic[2]
        return m_rd, std_rd, md_rd, gc_mrd_table


"""
fit the data, GC and NB
"""
def visal_rd_genome(bk_dataPath, fitDist = True):

    print("[Genome]: ** Generate basic statistics, Depth-of-Coverage hist ... ")
    # draw a read distribution 
    with h5py.File(bk_dataPath, 'r') as hf:
        
        tmpVec = np.array(hf["rdVec"][:])
        gc_list = hf["gc_list"][:]
        mrd_list = hf["mrd_list"][:]
        
        rdVec = []
        for i in range(len(tmpVec)):
            rdVec.extend(tmpVec[i])

        rdVec = np.array(rdVec)
        
        fig = plt.figure()
        plt.hist(rdVec, color="red", density= True, bins=range(0,int(np.median(rdVec)*5)+1,1))
        sampleName = os.path.basename(bk_dataPath)
        figName = "../experiment/data_profile/genome_bin_rd_hist-"+sampleName + ".png"
        

        # fit the data
        ## 1. norm distribution
        mu, std = ss.norm.fit(rdVec)
        logger.info("Cached %f Genome Depth-of-coverage statistics is [%f, %f]" %(config.DATABASE["genomeSampleRate"], mu, std))
        print("[Genome]: ** Cached %f Genome Depth-of-coverage statistics is [%f, %f]" %(config.DATABASE["genomeSampleRate"], mu, std))
        
        if fitDist == False:
            plt.close("all")
            return(0)

        x = np.linspace(0, int(np.median(rdVec)*5), int(np.median(rdVec)*5)+1)
        p = ss.norm.pdf(x, mu, std)
        norm_dist, = plt.plot(x, p, 'k', linewidth=2)
        
        ## 2. fit poission
        rv = ss.poisson(mu)
        poiss_dist, = plt.plot(x, rv.pmf(x), linewidth=2, color="blue")

        ## 3. fit negative binom, this step is slow
        param = fit_nbinom(rdVec)
        nb_dist, = plt.plot(x, ss.nbinom.pmf(x, param["size"], param["prob"]), linewidth=2, color="green")
        plt.legend([norm_dist, poiss_dist, nb_dist], ['norm', 'poission', 'negative binomial'], loc="upper right")        
        plt.savefig(figName)
        plt.close("all")


        # plot the GC
        fig = plt.figure()
        plt.scatter(gc_list, mrd_list, color="blue")
        plt.xlabel("GC")
        plt.ylabel("median RD")
        figName = "../experiment/data_profile/"+"GCscatter-"+sampleName + ".png"
        plt.savefig(figName)
        plt.close("all")



def get_NegBinomial_params(bk_dataPath):

    # draw a read distribution 
    with h5py.File(bk_dataPath, 'r') as hf:
        
        tmpVec = np.array(hf["rdVec"][:])
        gc_list = hf["gc_list"][:]
        mrd_list = hf["mrd_list"][:]
        
        rdVec = []
        for i in range(len(tmpVec)):
            rdVec.extend(tmpVec[i])

        param = fit_nbinom(rdVec)

        return (param["size"], param["prob"])


"""
if __name__ == "__main__":

    
    lowBamFile="/home/dev/NA12878.mapped.ILLUMINA.bwa.CEU.low_coverage.20121211.bam"
    highBamFile="/home/dev/NA12878.mapped.ILLUMINA.bwa.CEU.high_coverage_pcr_free.20130906.bam"
    uhighBamFile="/data/Dataset/GIAB/JP201712_data/HG001_NA12878/HG001.hs37d5.300x.bam"

    getSampleBG(lowBamFile, sampleFold=0.05)
"""


