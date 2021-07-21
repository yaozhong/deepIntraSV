"""
Description: transform A/T/G/C sequence vectors to be processed by defined filters
Data: 2017-06-29
Author: Yao-zhong Zhang@IMSUT
"""
from util import *

# inital version: used as the input data transormer for the input data 
# next improvement

def kmer_dic_gen():

	k = config.DATABASE["max_kmer"]
	#print("Initlaization %d k-mer word dictionary!" %(4**k))
	basic="atgc"
	kmers=['n']
	kmer2idx={}

	tmp = product(basic, repeat=k)
	mers = [ ''.join(x) for x in tmp ]
	kmers.extend(mers)

	# generate dictionary
	for i in range(len(kmers)):
		kmer2idx[ kmers[i] ] = i
		kmer2idx[ kmers[i].upper() ] = i

	return (kmer2idx, kmers)

def seq2vec(seq):

	kmerDic = config.DATABASE["kmer_dic"]
	k = config.DATABASE["max_kmer"]
	xlen = int(len(seq)/k)
	#mat = np.zeros(xlen, dtype=np.uint16 )
        mat = [0]*xlen

	if config.DATABASE["max_kmer"] > 1:
		for i in range(xlen):
			mat[i] = kmerDic.get(seq[i*k:(i+1)*k], 0)
	else:
		for i, nc in enumerate(seq):
			mat[i] = kmerDic.get(nc, 0)
	return mat

# this function is used when the memeory is limited
def rg2vec(rg, handle):

	# from hanle to get the region
	seq = handle[rg[0]][rg[1]:rg[2]]
	kmerDic = config.DATABASE["kmer_dic"]

	k = config.DATABASE["max_kmer"]
	xlen = int(len(seq)/k)
	mat = np.zeros( xlen, dtype=np.uint16 ) 

	# this is the time consumming part
	## when k > 1, the loop-based implemenation is much slower
	if config.DATABASE["max_kmer"] > 1:
            for i in range(xlen):
                mat[i] = kmerDic.get(seq[i*k:(i+1)*k], 0)
	else:
            for i, nc in enumerate(seq):
                mat[i] = kmerDic.get(nc, 0)
	return mat

def set_kmer_dic():
	
	kmer2idx, kmers = kmer_dic_gen()
	config.DATABASE["kmer_dic"] = kmer2idx


def TEST_case():

	set_kmer_dic()
	seq = "atcggcattagcctatccatgnaat"
	print seq2vec(seq)
	
if __name__ == "__main__":
	TEST_case()

