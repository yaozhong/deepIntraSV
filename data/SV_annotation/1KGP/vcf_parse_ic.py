"""
VCF-parse for getting related CNV break points of related sample
Date: 2018-11-19, 2018-11-26
"""

import matplotlib
matplotlib.use('Agg')  # this need for the linux env
import matplotlib.pyplot as plt

import vcf
import numpy as np
import time, argparse
from tqdm import tqdm
import json


# CPU version
def sample_sv_list_gen(vcf_file, highConfidenceThreshold=50, savePath="./", verbose=True):

    if verbose: start_time = time.time()
    #Get the correct VCF file from 1000GP Phase3
    vcf_reader = vcf.Reader(open('./ALL.wgs.mergedSV.v8.20130502.svs.genotypes.vcf', 'r'))
    count_dic = {}

    noEND_type_list = []
    num_inPrecision = 0
    
    sample_names = vcf_reader.samples
    sample_sv_dic = {}
    # generate the sample sv list dictionary.
    for sample in sample_names:
        sample_sv_dic[sample] = []

    print "* Start parsing VCF:" + vcf_file

    start_ci_range_list, end_ci_range_list = [], []  
    start_ci_list, end_ci_list = [], []
    record_ci, record_non_ci = 0,0

    # main loop go through all the samples
    for record in tqdm(vcf_reader):

        # skip inPrecision 
        if record.is_sv_precise == False:
            num_inPrecision += 1
            continue

        start = record.POS
        chrom = record.CHROM
    
        sv_type = record.INFO["SVTYPE"]
        sv_type_ALT = record.ALT

        # record basic statistic summary
        if sv_type in count_dic.keys():
            count_dic[sv_type] += 1
        else:
            count_dic[sv_type] = 1
    
        # extract candidate location
        try:
            end = record.INFO["END"]

            try:
                ci_start = record.INFO["CIPOS"]
                ci_end = record.INFO["CIEND"]

                ci_range_start = np.abs(ci_start[1] - ci_start[0])
                ci_range_end = np.abs(ci_end[1] - ci_end[0]) 
                
                start_ci_range_list.append(ci_range_start)
                end_ci_range_list.append(ci_range_end)

                start_ci_list.extend(ci_start)
                end_ci_list.extend(ci_end)

                rg = (chrom, start, end, sv_type, ci_start[0], ci_start[1], ci_end[0], ci_end[1])
                record_ci += 1
            except:
                rg = (chrom, start, end, sv_type, 1, -1, 1, -1)
                record_non_ci += 1

            het_sample = record.get_hets()
            hom_alts_sample = record.get_hom_alts()

            if record.num_het > 0:
                for s in het_sample:
                    sample_sv_dic[s.sample].append(rg)
                
            if record.num_hom_alt > 0:
                for s in hom_alts_sample:
                    sample_sv_dic[s.sample].append(rg)

        except:
            if sv_type not in noEND_type_list:
                noEND_type_list.append(sv_type)

    print "* VCF parse done!"
    print "** noEND types"
    print noEND_type_list

    sample_sv_count = [ len(sample_sv_dic[sn]) for sn in sample_names  ]
    print("-------- Sample VCF statsitics for CI-%d ----------- " %(highConfidenceThreshold))
    print("NA12878: %d " %(len(sample_sv_dic["NA12878"])))
    print("minmal=%d, maximum=%d, mean=%d, std=%d" %(np.min(sample_sv_count), np.max(sample_sv_count), np.mean(sample_sv_count), np.std(sample_sv_count)))
    print "-"*50

    print("In total record, IC information containing record %d, no containing %d" %(record_ci, record_non_ci))

    if verbose:
        used_time = time.time() - start_time
        print("=Total used %s seconds=" %(used_time))

        # draw the histgram
        fig = plt.figure()
        plt.hist(start_ci_range_list, color="blue", density=False, bins=range(np.min(start_ci_range_list), np.max(start_ci_range_list), 1))
        figName = "./start-ic-range_hist.png"
        plt.savefig(figName)
        plt.close("all")


        fig = plt.figure()
        plt.hist(end_ci_range_list, color="blue", density=False, bins=range(np.min(end_ci_range_list), np.max(end_ci_range_list), 1))
        figName = "./end-ic-range-hist.png"
        plt.savefig(figName)
        plt.close("all")


        fig = plt.figure()
        plt.hist(start_ci_list, color="blue", density=False, bins=range(np.min(start_ci_list), np.max(start_ci_list), 1))
        figName = "./start-ic-hist.png"
        plt.savefig(figName)
        plt.close("all")


        fig = plt.figure()
        plt.hist(end_ci_list, color="blue", density=False, bins=range(np.min(end_ci_list), np.max(end_ci_list), 1))
        figName = "./end-ic-hist.png"
        plt.savefig(figName)


    # caching data
    print("* Cache the file to " + savePath)
    jd = json.dumps(sample_sv_dic)
    output = open(savePath+"/sample_sv_IC-"+str(highConfidenceThreshold)+".json", "w")
    output.write(jd)
    output.close()
    
def load_sample_sv(filePath):

    with open(filePath, "r") as f:
        
        sample_sv_dic = json.load(f)
        return sample_sv_dic

def genAnnoFile(sname, sample_sv_dic, ci):
    
    #print("\n* %s has %d SVs" %(sname, len(sample_sv_dic[sname])))

    svs = sample_sv_dic[sname]

    output_sv = "annoFile/"+ sname + "_1kGP_IC-" + str(ci)+".bed"
    file_output = open(output_sv, "w")
    num_del, num_dup, num_cnv, num_ins, num_other = 0, 0, 0, 0, 0
    file_output.write("#chr\tstart\tend\ttype\tstart_ci_l\tstart_ci_r\tend_ci_l\tend_ci_r\n")

    num_sv = 0

    for sv in svs:
        if sv[0] == "X" or sv[0] == "Y":
            continue

        if sv[2] - sv[1] < 50:
            continue

        num_sv += 1
    
        if sv[3] == "DEL" or sv[3] =="DEL_ALU" or sv[3]=="DEL_LINE1" or sv[3]=="DEL_SVA": num_del +=1
        elif sv[3]== "CNV": num_cnv +=1
        elif sv[3] == "DUP": num_dup +=1
        elif sv[3] == "INS": num_ins += 1
        else: num_other += 1

        file_output.write("%s\t%d\t%d\t%s\t%d\t%d\t%d\t%d\n" %(sv[0], sv[1], sv[2], sv[3], sv[4], sv[5], sv[6], sv[7]))

    print("\n* %s has %d SVs" %(sname, num_sv))
    print("%s: DEL=%d, DUP=%d, CNV=%d, INS=%d, OTHER=%d" %(sname, num_del, num_dup, num_cnv, num_ins, num_other))
    file_output.close()
    return svs


def compareOverlap(na12878, sample, sname, ci):    

    output_sv = "annoFile/nonOverlap_"+ sname + "_1kGP_IC-" + str(ci)+".bed"
    overlap_sv = "annoFile/overlap_"+ sname + "_1kGP_IC-" + str(ci)+".bed"

    file_output = open(output_sv, "w")
    file_overlap = open(overlap_sv, "w")

    file_output.write("#chr\tstart\tend\ttype\tstart_ci_l\tstart_ci_r\tend_ci_l\tend_ci_r\n")
    file_overlap.write("#chr\tstart\tend\ttype\tstart_ci_l\tstart_ci_r\tend_ci_l\tend_ci_r\n")

    overlap, nonOverlap = 0, 0
    for sv in sample:

        if sv[0] == "X" or sv[0] == "Y":
            continue

        if sv[2] - sv[1] < 50:
            continue

        if sv in na12878:
            overlap += 1
            file_overlap.write("%s\t%d\t%d\t%s\t%d\t%d\t%d\t%d\n" %(sv[0], sv[1], sv[2], sv[3], sv[4], sv[5], sv[6], sv[7]))
        else:
            nonOverlap += 1
            file_output.write("%s\t%d\t%d\t%s\t%d\t%d\t%d\t%d\n" %(sv[0], sv[1], sv[2], sv[3], sv[4], sv[5], sv[6], sv[7]))

    print("* %s has %d-overlap, %d-nonOverlap" %(sname, overlap, nonOverlap))
    file_output.close()
    file_overlap.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser("Parse vcf file")
    parser.add_argument('--ci', '-i', type=int, default=50, help="Confidence interval threshold.-1 represent no filtering")
    parser.add_argument('--savePath', '-s', type=str, default='./', help="Saving the path of the file")
    parser.add_argument('--mode', '-m', type=str, default='parse', help="parse/analysis")
    args= parser.parse_args()
    
    vcf_file = './ALL.wgs.mergedSV.v8.20130502.svs.genotypes.vcf'
    if args.mode == "parse":
        assert(args.ci == -1)
        sample_sv_list_gen(vcf_file, args.ci, args.savePath)

    if args.mode == "analysis":

        cache_file = args.savePath+"/sample_sv_IC-"+str(args.ci)+".json"
        sample_sv_dic = load_sample_sv(cache_file)

        sample_list = ["NA12878", "NA12156", "NA19238", "NA19239", "HG00513", "HG00731", "HG00419", "NA18939"]
        sv_list = []       
        for sname in sample_list:
            sample_sv = genAnnoFile(sname, sample_sv_dic, args.ci)
            sv_list.append(sample_sv)

        for i in range(1, len(sv_list)):
            compareOverlap(sv_list[0], sv_list[i], sample_list[i], args.ci)



        

