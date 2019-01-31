"""
Data:	2018-05-29
Author: Yao-zhong Zhang @ IMSUT

Description:
Basic functions for read region information

# revised for the chrName normalization part
"""
from __future__ import division

import logging, config
import time, os, random, sys
import re, random
import h5py

import pysam, pyfaidx, pyBigWig

from rpy2.robjects.packages import importr
import rpy2.robjects as robjects

import matplotlib
matplotlib.use('Agg')  # this need for the linux env
import matplotlib.pyplot as plt

from itertools import product

import numpy as np
import multiprocessing

import argparse
import tensorflow as tf
from keras import backend as K

## this part for the reproducible of the CPU version
print "@ Setting randomness fixed ...."
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(config.DATABASE["rand_seed"])
random.seed(config.DATABASE["rand_seed"])
tf.set_random_seed(config.DATABASE["rand_seed"])

"""
# uncomment when to reproduce the results in CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
"""

print "- Random seed set [DONE]!! (* not work for the GPU training)\n"

date_stamp = time.strftime("%Y%m%d_%H.%M")

## setting the log information
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create a file hander
handler = logging.FileHandler(filename='../experiment/LOG/log_'+ date_stamp +".txt")
handler.setLevel(logging.INFO)
# create logging format
formatter = logging.Formatter('[%(asctime)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
# add handlers to the log:ger
logger.addHandler(handler)

handler2 = logging.FileHandler(filename='../experiment/LOG/debug_'+ date_stamp + ".txt")
handler2.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler2.setFormatter(formatter)
logger.addHandler(handler2)


def get_chr_region(filePath):
	rgs = []

	with open(filePath) as f:
		for line in f:
			elems = line.strip().split("\t")
			rgs.append((elems[0], int(0), int(elems[1])))

	return rgs


def get_blackList_regions():

	rgs = []
	with open(config.DATABASE["black_list"]) as f:
		for line in f:
			L = line.strip().split()

			if(len(L) != 3):
				print "[Warning]: Please check the format of black list files!!"
				exit(-1)

			rgs.append((str(L[0]), int(L[1]), int(L[2])))
	
	return rgs

###############################################################	

def normChrName(rg, withChr=0):
 
    chrName = str(rg[0]).lower()  
    if re.search("chr", chrName) is not None:
        if withChr == 0:
		    chrName = chrName.replace("chr", "")
    else:
	    if withChr != 0:
	        chrName = "chr" + chrName
    
    rg = list(rg)
    rg[0] = chrName
    
    return tuple(rg)


def normChrStrName(chrName, withChr=0):
	
        chrName = str(chrName).lower()

	if re.search("chr", chrName) is not None:
		if withChr == 0:
			chrName = chrName.replace("chr", "")
	else:
		if withChr != 0:
			chrName = "chr"+chrName

	return chrName



######################################################
def estimate_sampleSize(binSize=1000):

	print("== Estimate the sample size if doing bining %d ==" %(binSize))
	rgs = get_chr_region(config.DATABASE["chrLen_info"])
	total_nc = 0
	chr_bp = 0
	for rg in rgs:
		total_nc += rg[2]
		chr_bp.append(rg[2])

	print("* Original nucleotide %d bp" %(total_nc))
	print ("* After bin sizing %d bins" %(total_nc/binSize))
	print chr_bp/total_nc
	print("===================================================")


def estimate_sampleSize(binSize=1000):

	print("== Estimate the sample size if doing bining %d ==" %(binSize))
	rgs = get_chr_region(config.DATABASE["chrLen_info"])
	total_nc = 0
	chr_bp = {}
	for rg in rgs:
		total_nc += rg[2]
		chr_bp[rg[0]] = rg[2]

	print("* Original nucleotide %d bp" %(total_nc))
	print ("* After bin sizing %d bins" %(total_nc/binSize))
	print("--------------------------------------------------")
	print "Chr\tPercentage\tCum-percentage "
	print("--------------------------------------------------")
	cum = 0
	for chrome, per in sorted(chr_bp.items(), key=lambda kv:kv[1], reverse=True ):
		cum += per/total_nc * 100
		print chrome, "\t", np.around(per/total_nc * 100, 2), "\t", np.around(cum, 2)  
	print("===================================================")



def check_all_zero(x):
    total = np.sum(x)
    if total == 0:
        return True
    return False



if __name__ == "__main__":
	estimate_sampleSize(100)

