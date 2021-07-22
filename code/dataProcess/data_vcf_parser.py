"""
VCF-parse for getting related CNV break points of related sample
Date: 2018-11-19, 2018-11-26, 2019-12-22
"""

import matplotlib
matplotlib.use('Agg')  # this need for the linux env
import matplotlib.pyplot as plt
#from data_bam2rd import get_RDVec_parallel_nobin

import vcf
import numpy as np
import time, argparse
import json
import pandas as pd
from util import *

from sklearn.model_selection import StratifiedShuffleSplit

def filter_ci(rg):
    left_ci_length, right_ci_length = rg[5]-rg[4], rg[7]-rg[6]
    if left_ci_length > config.DATABASE["vcf_ci"] or right_ci_length > config.DATABASE["vcf_ci"]:
        return False
    else:
        return True

"""
2019-11-18
For simulation region, no confidence region is required
Return: return random split pandas dataframe of train-set and test-set
"""
# currently this is only used for the simulation data
def parse_sim_data_vcf(vcf_file, test_protion=0.2, verbose=True, vcf_filter=False):

    if verbose: start_time = time.time()

    head_name = ['chr', 'start', 'end', 'sv_type', 'left_ci_1', 'left_ci_2', 'right_ci_1', 'right_ci_2']
    long_read_source_count = 0
    target_chr = [str(i) for i in range(1,23)]

    # add the indication of the filtering of the confidence interval
    ci_filtered_rg_num = 0
    max_hg2_count = []

    # parsing the bed file
    if vcf_file.endswith("bed"):
        print("[*] Start parsing BED:" + vcf_file)
        df = pd.read_csv(vcf_file, sep="\t", header=0, names=head_name)
        df.chr = df.chr.apply(str)

        # 2020-03-13 correct the inconsistence of vcf and bed file
        #  Bed file is tranformed from VCF (1-based), need to transformed to 0-based for using pysam.
        df.start = df.start - 1

    if vcf_file.endswith("vcf"):
        vcf_reader = vcf.Reader(open(vcf_file, 'r'))
        sv_range_list = []
        print("[*] Start parsing VCF:" + vcf_file)
    
        for record in tqdm(vcf_reader):

            # only using the PASS VCF PASS=[], HG00514 no filter note!!
            # only applys to VCF input
            if (vcf_filter):
                if len(record.FILTER) > 0:
                    continue

            try:
                # note VCF is 1-based system, fully closed.
                # pysam is 0-based , halp open.
                start = record.POS - 1 
                chrom = record.CHROM

                if config.DATABASE["chr_prefix"]:
                    chrom = "chr"+chrom

                sv_type = record.INFO["SVTYPE"]
                if(isinstance(sv_type, list)): sv_type = sv_type[0]
                sv_type_ALT = record.ALT
                
                # for Riken's VCF NA12878
                try:
                    if record.INFO["SOURCE"] == "LONG_READ":
                        #print("*filter out long read ...")
                        long_read_source_count += 0
                        continue
                except:
                    pass

                # filtering out GRCh38 imprecision data for the training.
                try:
                    if(record.INFO["IMPRECISE"]):
                        continue
                except:
                    pass
  
  
                try:
                    max_hg2_count.append(record.INFO["HG2count"])
                    if(record.INFO["HG2count"] < 8):
                        continue
                except:
                    pass

                try:
                    end = record.INFO["END"]
                    if(isinstance(end, list)): end = end[0]
                    svlen = end - start    
                except:
                    #continue
                    svlen = record.INFO["SVLEN"]
                    if(isinstance(svlen, list)): svlen = svlen[0]
                    end = start + np.abs(svlen)
                    
                # note the order of end, keep the order
                try:
                    rg = (chrom, start, end, sv_type, \
                        record.INFO["CIPOS"][0], record.INFO["CIPOS"][1],\
                        record.INFO["CIEND"][0], record.INFO["CIEND"][1])
                except:
                    # note the defualt value for different VCF file.
                    # make the change to the unkonwn mode 20200224
                    rg = (chrom, start, end, sv_type, 1, -1, 1, -1)

                # filtering rg length, added 2020-02-17, note this part may not filtering HG002
                if rg[2] - rg[1] < 50: continue

                sv_range_list.append(rg)

                if verbose:
                    print(rg)

            except:
                if verbose:
                    print("[x] Reading erros happens for the record:")
                    print(record)

        df = pd.DataFrame(sv_range_list, columns=head_name)

    if len(max_hg2_count) > 0:
        print("=========== HG002 statisics =============")
        print(np.mean(max_hg2_count), np.min(max_hg2_count), np.max(max_hg2_count))
        print( np.percentile(max_hg2_count, 75), np.percentile(max_hg2_count, 50), np.percentile(max_hg2_count, 25))

    print("\n[ci_filter]: Before filtering,  [%d] rgs" %(df.shape[0]))
    df = df[df.apply(filter_ci, axis=1)]
    print("[ci_filter]: After filtering rgs with confidence interval larger than ci=[%d], [%d] rgs" %(config.DATABASE["vcf_ci"], df.shape[0]))
    df.start = df.start.apply(int)
    df.end = df.end.apply(int)

    sv_type_group = df.groupby("sv_type").count()
    print(sv_type_group.iloc[:,0])
    
    #split the range data for training and testing
    sv_types = sv_type_group.index.tolist()
    print("[*] Having %d types of SVs: " %(len(sv_types))),
    print(sv_types)

    # stratify shuff split
    if test_protion > 0:
        sample = StratifiedShuffleSplit(n_splits=1, test_size=test_protion, random_state=len(sv_types))

        for train_idx, test_idx in sample.split(df, df["sv_type"]):
            train_sv = df.loc[train_idx]
            test_sv = df.loc[test_idx]

        if verbose:
            used_time = time.time() - start_time
            print("* Parse VCF/BED totally used time %f" %(used_time))
    else:
        train_sv = df
        test_sv = pd.DataFrame()

    ## transform pd to list
    train_rgs, test_rgs, all_rgs = [], [], []

    for index, sv in train_sv.iterrows():
        train_rgs.append(sv)

    if not test_sv.empty:
        for index, sv in test_sv.iterrows():
            test_rgs.append(sv)

    for index, sv in df.iterrows():
        # checking the range format
        all_rgs.append(sv)

    print("[*] Splited of the data size is Train=%d, Test=%d, All=%d" %(len(train_rgs), len(test_rgs), len(all_rgs)))
    print("[-] filter out long read source SV of [%d].\n" %(long_read_source_count))

    return train_rgs, test_rgs, all_rgs

##########################################################

# 2019-07-18
def sample_sv_list_gen(vcf_file, highConfidenceThreshold=50, savePath="./", verbose=True):

    if verbose: start_time = time.time()

    #Get the correct VCF file from 1000GP Phase3
    vcf_reader = vcf.Reader(open(vcf_file, 'r'))
    count_dic = {}

    noEND_type_list = []
    num_inPrecision = 0
    
    sample_names = ["riken_sim_A"]
    sample_sv_dic = {}
    # generate the sample sv list dictionary.
    for sample in sample_names:
        sample_sv_dic[sample] = []

    print("* Start parsing VCF:" + vcf_file)

    start_ci_range_list, end_ci_range_list = [], []  
    start_ci_list, end_ci_list = [], []
    record_ci, record_non_ci = 0,0

    # main loop go through all the samples
    for record in tqdm(vcf_reader):

        try:
            start = record.POS
            chrom = record.CHROM
            sv_type = record.INFO["SVTYPE"]
            sv_type_ALT = record.ALT

            # extract the length
            svlen = record.INFO["SVLEN"]
            end = start + np.abs(svlen)

            #if svlen < 0:
            #    start, end = end, start
        
            # note the order of end, keep the order
            rg = (chrom, start, end, sv_type, 1, -1, 1, -1)
            sample_sv_dic[sample_names[0]].append(rg)
        except:
            record_non_ci += 1

    # writing to the new annotation file
    svs = sample_sv_dic[sample_names[0]]

    output_sv = vcf_file +".bed"

    file_output = open(output_sv, "w")
    num_del, num_dup, num_cnv, num_ins, num_other = 0, 0, 0, 0, 0
    file_output.write("#chr\tstart\tend\ttype\tstart_ci_l\tstart_ci_r\tend_ci_l\tend_ci_r\n")

    num_sv = 0

    for sv in svs:
        if sv[0] == "X" or sv[0] == "Y":
            continue

        #if sv[2] - sv[1] < 50:
        #    continue

        num_sv += 1
    
        # changing the code for this part that can automatically 
        if sv[3] == "DEL" or sv[3] =="DEL_ALU" or sv[3]=="DEL_LINE1" or sv[3]=="DEL_SVA": num_del +=1
        elif sv[3]== "CNV": num_cnv +=1
        elif sv[3] == "DUP": num_dup +=1
        elif sv[3] == "INS": num_ins += 1
        else: num_other += 1

        file_output.write("%s\t%d\t%d\t%s\t%d\t%d\t%d\t%d\n" %(sv[0], sv[1], sv[2], sv[3], sv[4], sv[5], sv[6], sv[7]))

    print("\n* %s has %d SVs" %(sample_names[0], num_sv))
    print("%s: DEL=%d, DUP=%d, CNV=%d, INS=%d, OTHER=%d" %(sample_names[0], num_del, num_dup, num_cnv, num_ins, num_other))
    file_output.close()
    return svs


# 2019-11-21 generate input list for the vcf list
def gen_input_for_vcf(vcf_file, bam_file, bin=100):

    print("* VCF data split ...")
    train_sv, test_sv = parse_sim_data_vcf(vcf_file, 0.4, False)
    print(" |- train_sv has %d SVs" %(len(train_sv.index)))
    print(" |- test_sv has %d SVs" %(len(test_sv.index)))

    print("* Generation the test file ...")
    rgs = []
    for index, sv in test_sv.iterrows():

        # checking the range format
        rg = (sv["chr"], sv["start"] - bin/2, sv["start"] + bin/2, sv["sv_type"]) #, "L", bin/2, 0, 0)
        rgs.append(rg)

    rdVec = get_RDVec_parallel_nobin(bam_file, rgs)

    print(rdVec)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Parse Riken vcf file format test...")
    parser.add_argument('--vcf', '-v', type=str, default="", required=True, help="VCF input ")
    parser.add_argument('--output', '-o', type=str, default="", required=True, help="VCF input ")

