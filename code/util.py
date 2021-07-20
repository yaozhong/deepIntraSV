"""
Data:	2018-05-29
Author: Yao-zhong Zhang @ IMSUT

Description:
Basic functions for read region information

# revised for the chrName normalization part
"""
from __future__ import division

import config
import time, os, random, sys
import re, random
import h5py
from tqdm import tqdm

import pysam, pyfaidx, pyBigWig

import matplotlib
matplotlib.use('Agg')  # this need for the linux env
import matplotlib.pyplot as plt

from itertools import product

import numpy as np
import multiprocessing

import argparse
import tensorflow as tf
from keras import backend as K
from datetime import date, datetime
import pickle

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")

## this part for the reproducible of the CPU version
date_stamp = time.strftime("%Y%m%d_%H.%M")
print("\n============================================")
print("| Intra-bin break-point segmentation v0.1  |")
print("============================================\n")
print("* Initialization started"),
print(time.strftime("%Y/%m/%d_%H:%M"))

print "* Try to fix potential randomness ... "

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

print("- Random seed set for np.random, random, tf_random with seed [%d]\n" %(config.DATABASE["rand_seed"]))

## setting the log information
"""
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
"""


def get_chr_region(filePath):
	rgs = []

	with open(filePath) as f:
		for line in f:
			elems = line.strip().split("\t")
			rgs.append((elems[0], int(0), int(elems[1])))

	return rgs

def get_chr_length(filePath):
	len_dic = {}
	with open(filePath) as f:
		for line in f:
			elems = line.strip().split("\t")
			chrName = str(elems[0].replace("chr", ""))

			if config.DATABASE["chr_prefix"] == True:
				chrName = "chr"+chrName

			len_dic[chrName] = int(elems[1])

	return len_dic

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


# calcuate consectutive labels 
def get_break_point_position(pred):
	idx = 0
	seq, count = [], []
	seq_len = len(pred)

	while(idx < seq_len):

		current = pred[idx]
		tmp_count = 1

		while( idx + 1 < seq_len and pred[idx+1] == current ): 
			idx += 1
			tmp_count += 1
		
		seq.append(current)
		count.append(tmp_count)
		idx += 1

	return seq, count, np.cumsum(count)


def genWeightMatrix(hot_rgs):

	n_sample= len(hot_rgs)
	weights = np.zeros((n_sample, config.DATABASE["binSize"]))

	for i in range(n_sample):
		for j in range(hot_rgs[i][0], hot_rgs[i][1]):
			weights[i,j] = 1

	return weights


# added 2020-1-18 add training files, save in the bed format
def save_train_rgs(rgs, save_file):

	now = datetime.now()
	dt_string = now.strftime("%Y_%m_%d-")

	save_file=save_file + dt_string + config.DATABASE["model_data_tag"] +"-train_rgs.txt"

	with open(save_file, "wb") as fp:
		pickle.dump(rgs, fp)
		print("[I/O]: Trained rgs is saved in [%s]" %(save_file))


def load_train_rgs(save_file):

	if not os.path.exists(save_file):
		return None

	with open(save_file, "rb") as fp:
		rgs = pickle.load(fp)
		return rgs


def get_split_len_idx(sv_list):

	dic = {}
	dic["50-99"], dic["100-199"], dic["200-299"], dic["300-499"], dic["500-999"], dic["1000-"] = [],[],[],[],[],[]
	sv_range = ["50-99", "100-199", "200-299", "300-499", "500-999", "1000+"]
	sv_count = [0,0,0,0,0,0]

	for i in range(len(sv_list)):
		sv_len= int(sv_list[i][2]) - int(sv_list[i][1])

		if sv_len < 50 : continue

		if sv_len < 100 :
			dic["50-99"].append(i)
			sv_count[0] += 1
		elif sv_len < 200:
			dic["100-199"].append(i)
			sv_count[1] += 1
		elif sv_len < 300:
			dic["200-299"].append(i)
			sv_count[2] += 1
		elif sv_len < 500:
			dic["300-499"].append(i)
			sv_count[3] += 1
		elif sv_len < 1000:
			dic["500-999"].append(i)
			sv_count[4] += 1
		else:
			dic["1000-"].append(i)
			sv_count[5] += 1
	
	return sv_range, sv_count

# added 2020-02-09 added split the SVs accroding to length
def get_split_len_idx2(sv_list):

	dic = {}
	dic["50-99"], dic["100-199"], dic["200-299"], dic["300-499"], dic["500-999"], dic["1000-"] = [],[],[],[],[],[]

	for i in range(len(sv_list)):
		sv_len= int(sv_list[i][2]) - int(sv_list[i][1])

		if sv_len < 50 : continue

		if sv_len < 100 :
			dic["50-99"].append(i)
		elif sv_len < 200:
			dic["100-199"].append(i)
		elif sv_len < 300:
			dic["200-299"].append(i)
		elif sv_len < 500:
			dic["300-499"].append(i)
		elif sv_len < 1000:
			dic["500-999"].append(i)
		else:
			dic["1000-"].append(i)
	return dic

# get the dist idx
def get_dist_len_idx(dist):

	if dist < 5:   return 0
	if dist < 10:  return 1
	if dist < 20:  return 2
	if dist < 50:  return 3
	if dist < 100: return 4
	if dist < 200: return 5
	if dist < 500: return 6
	if dist < 1000: return 7
	return 8


# added 2020-02-17, checking the whehter the boundary are too close
def filtering_boundary_too_close_SVs(rgs):

	chr_len = get_chr_length(config.DATABASE["chrLen_info"])
	filter_rg = []
	flank = int(config.DATABASE["binSize"]/2)

	for rg in rgs:
		if (rg[1]- flank >0) and (rg[2]+flank < chr_len[str(rg[0])]):
			filter_rg.append(rg)

	return filter_rg

# 2020-02-27
def vcf2bed(vcf_file, output_file):

    _, _, svs = parse_sim_data_vcf(vcf_file, 0, False, False)

    file_output = open(output_file, "w")
    
    for sv in svs:
        file_output.write("%s\t%d\t%d\t%s\n" %(sv[0], sv[1], sv[2], sv[3]))

    file_output.close()


if __name__ == "__main__":
	pred = "0000000111111000000"
	print(pred)
	seq, seq_count, index = get_break_point_position(pred)
	print(seq)
	print(seq_count)
	print(index)
