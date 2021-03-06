"""
Date: 2018-06-10
Author: Yao-zhong Zhang
Description:
GC content calculation for the given region and basic parallel implementation test
"""
from __future__ import division
from util import *
from data_seq2feat import *
import traceback 
from data_feat_Fragility import *

# basic CPU version of calculating GC content with larger content
def getRegion_GC(region, binSize, fastaFile):

	with pyfaidx.Fasta(fastaFile, as_raw=True) as fa_file:
		nSeg = int((region[2] - region[1])/binSize)
		gc_vec = np.zeros(nSeg, np.float32)

		for i in range(nSeg):

			rg = (region[0], region[1] + i*binSize, region[1] + (i+1)*binSize)
                        seqs = fa_file[rg[0]][rg[1]:(rg[2])]
                        if len(seqs) < (rg[2] - rg[1]):
                            seqs = 'N'*(rg[2]-rg[1])
                
                        gc_vec[i] = calcualte_gc(seqs)

	return gc_vec


def getRegions_GC_nobin(regions, fastaFile):

	with pyfaidx.Fasta(fastaFile, as_raw=True) as fa_file:

		gc_vec = np.zeros(len(regions), np.float32)

		for rg in regions:
                    
		    seqs = fa_file[rg[0]][rg[1]:(rg[2])]
                    if len(seqs) < (rg[2] - rg[1]):
                        seqs = 'N'*(rg[2]-rg[1])
                        logger.debug(rg)

                    gc_vec[i] = calcualte_gc(seqs)

	return gc_vec



## current usage of extracting GC content
# key features extraction
# original GC and Seq feature extractor

def getRegions_GC_seq_worker(params):

	fastaFile, regions = params
	ext = config.DATABASE["extra_feat_ext_len"]

	with pyfaidx.Fasta(fastaFile, as_raw=True, one_based_attributes=False) as fa_file:
		
		gc_vec = np.zeros(len(regions), np.float32)
		seq_vec = []
		flex_vec, stable_vec =[], []
		for i, rg in enumerate(regions):

			seqs = fa_file[rg[0]][rg[1]:(rg[2]+ext)]
			if len(seqs) < (rg[2]+ext -rg[1]):
				seqs = 'N'*(rg[2]+ext -rg[1])
				logger.debug(rg)

			# addtional features added here
			gc_vec[i] = calcualte_gc(seqs)
			seq_vec.append(seq2vec(seqs[:(len(seqs)-ext)]))

			# adding addtional vec stability information.
			flex_vec.append(calc_flexibility_vec(seqs))
			stable_vec.append(calc_stability_vec(seqs))

		return (gc_vec, np.array(seq_vec, dtype=np.uint16), \
				np.array(flex_vec, dtype=np.float16), np.array(stable_vec, dtype=np.float16))

"""	    
def getRegions_GC_seq_worker(params):

	fastaFile, regions = params

	with pyfaidx.Fasta(fastaFile, as_raw=True, one_based_attributes=False) as fa_file:

		gc_vec = np.zeros(len(regions), np.float32)
		seq_vec = []

		for i, rg in enumerate(regions):
			seqs = fa_file[rg[0]][rg[1]:(rg[2])]
			if len(seqs) < (rg[2] -rg[1]):
				seqs = 'N'*(rg[2]-rg[1])
				logger.debug(rg)

			gc_vec[i] = calcualte_gc(seqs)
			seq_vec.append(seq2vec(seqs))

		return (gc_vec, np.array(seq_vec, dtype=np.uint16))
"""

def getRegions_GC_worker(params):
	
	fastaFile, regions = params
	
	with pyfaidx.Fasta(fastaFile, as_raw=True) as fa_file:
		
		gc_vec = np.zeros(len(regions), np.float32)
		for i, rg in enumerate(regions):
                        seqs = fa_file[rg[0]][rg[1]:(rg[2])]
                        if len(seqs) < (rg[2] - rg[1]):
                            seqs = 'N'*(rg[2] - rg[1])
                            logger.debug(rg)

			gc_vec[i] = calcualte_gc(seqs)
			
	return gc_vec


def getRegions_GC_parallel_nobin(fastaFile, rgs):
	
	cores = multiprocessing.cpu_count()
	pool = multiprocessing.Pool(processes=cores)
	
	# core based split
	step = int(len(rgs)/cores)
	rgsList = [ rgs[i*step: (i+1)*step] for i in range(cores-1)]
	rgsList.append( rgs[(cores-1)*step:])
	
	todo_params = [(fastaFile, rs) for rs in rgsList ]
	output = pool.map(getRegions_GC_worker, todo_params)
	
	pool.close()
	pool.join()
	
	gc_vec = np.concatenate(output)
	return gc_vec


# parallel running 
def getRegions_GC_seq_parallel_nobin(fastaFile, rgs):

	if config.DATABASE["kmer_dic"] == None:
		set_kmer_dic()

	cores = multiprocessing.cpu_count()
	if len(rgs) < cores:
		cores = len(rgs)

	pool = multiprocessing.Pool(processes=cores)

	# core based split
	step = int(len(rgs)/cores)
	rgsList = [ rgs[i*step: (i+1)*step] for i in range(cores-1)]
	rgsList.append( rgs[(cores-1)*step:])

	todo_params = [(fastaFile, rs) for rs in rgsList ]
	output = pool.map(getRegions_GC_seq_worker, todo_params)
	
	pool.close()
	pool.join()

	# this part should be tested
	gc_vec = np.concatenate([x[0] for x in output])
	seqMat = np.concatenate( [ x[1] for x in output ])
	flexMat = np.concatenate( [ x[2] for x in output ])
	stableMat = np.concatenate( [ x[3] for x in output ])

	return (gc_vec, seqMat, flexMat, stableMat)



def getRegion_split(region, binSize):

	nSeg = int((region[2]-region[1])/binSize)
	rg_vec =[ (region[0], region[1]+i*binSize, region[1]+(i+1)*binSize) for i in range(nSeg) ]
	return rg_vec



def getRegion_GC_rg(region, binSize, fastaFile):

    with pyfaidx.Fasta(fastaFile, as_raw=True) as fa_file:
        
        nSeg = int((region[2] - region[1])/binSize)
        gc_vec = np.zeros(nSeg, np.float32)
        rg_vec = []

	for i in range(nSeg):
            rg = (region[0], region[1] + i*binSize, region[1] + (i+1)*binSize)
	    seqs = fa_file[rg[0]][rg[1]:(rg[2])]
	    gc_vec[i] = calcualte_gc(seqs)
	    rg_vec.append(rg)
			
	return (gc_vec, rg_vec)


def calcualte_gc(subseq):

	seq = subseq.lower()

	gc = seq.count('g') + seq.count('c')
	nc = seq.count('n')
	if nc == len(seq):
		return 0
	else:
		return gc/(len(seq) - nc)

#calculate the sequential GC shift
def calcualte_gc_vec(seq, rlen):
	vec = [ calcualte_gc(seq[i:i + rlen]) for i in range(len(seq)-rlen) ]
	return vec


##########################################################
# define test functions
##########################################################
def TEST_chr_cpu():

	faFile = "/data/Dataset/1000GP/reference/hs37d5.fa"

	hg_chr19_info = "./database/hg19_chr_info.txt"
	rgs = get_chr_region(hg_chr19_info)
	chr_gc_dic = {}

	start_time = time.time()

	for rg in rgs:
		tmp_time = time.time()
		chr_gcVec = getRegion_GC(rg, 500, faFile)
		logger.info("For Chr-%s, GC calculation used total %d seconds." %(rg[0], time.time()-tmp_time))
	logger.info("The whole chromesome totally used time is %d seconds." %(time.time()- start_time))



if __name__ == "__main__":
	#TEST_chr_cpu()


	fastaFile = "/data/Dataset/1000GP/reference/hs37d5.fa"
        regions = [('21', 14980939, 14981939)]       

	with pyfaidx.Fasta(fastaFile, as_raw=True) as fa_file:

		gc_vec = np.zeros(len(regions), np.float32)

		for rg in regions:
                    try:
			# seqs = fa_file[rg[0]][rg[1]:(rg[2])]
		        seqs = fa_file.get_seq(rg[0], rg[1]+1, rg[2])
                    except:
                        seqs = 'N'*(rg[2]-rg[1])

                    print seqs
                    print calcualte_gc(seqs)


	
