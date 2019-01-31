"""
Data:	2018-08-15
Author: Yao-zhong Zhang @ IMSUT

Description:
Break point data generation basedon on NA12878
validated GS data.

# 2018-08-22 re-editted version with data augmentation.
# 2018-08-24 revision on the background bin generation, not contain any already known break points. 
# 2018-09-19 revised the Y label to more detailed label(only using DEL and DUP in the gold standard data).
# 2018-10-17 Generate segment mask for training Unet
# 2018-10-21 fix a bug that rgs is not match after random shuffling
# 2018-11-05 clean the data for v1.0 release
# 2018-12-15 add the confidence interval for evluation
"""

import config

from util import *
from data_bam2rd import *
from data_GC import *
from data_Mappability import *
from data_loadCNVannotation import * 

"""
The function generate the vectors surrounding the break point.
According to the CNV size the context is also variant.
Usually larger copy number segements requires larger context.

"""

def input_gen_fromRgs(rgs, bamFilePath):

    rdVec = get_RDVec_parallel_nobin(bamFilePath, rgs)
    maVec = getRegions_mappability_parallel_nobin(config.DATABASE["mappability_file"], rgs)
   
    gcVec, seqMat= getRegions_GC_seq_parallel_nobin(config.DATABASE["ref_faFile"], rgs)

    return (rdVec,seqMat, maVec, gcVec)


"""
Checking the given region conains any bk point in the bk-region check list. 
"""
def whetherNotBKregion(region, bkDic):
    
    for bk in bkDic[region[0]]:
        if bk >= region[1] and bk < region[2]:
            return False
    return True

"""
Given Region return the y-mask, no change for the y, only the input.
"""

def getPairMask(cnvType, rg, shift, binSize):

	left, right = [0]*binSize, [0]*binSize
	rg_len = rg[2] - rg[1]
	
        if rg_len < binSize - shift:
		left[shift:(shift + rg_len)] = [1]*rg_len
	else:
		left[shift:binSize] = [1]*(binSize-shift) 
	
	if rg_len < shift:
		right[(shift-rg_len):shift] = [1]*rg_len
	else:
		right[0:shift] = [1]*shift

	return (left, right)


# submian function get the binSamples
def genData_from_rgLists(rgList, bkDic, bamFile, dataAug=0, bk_in=True):
	
	bk_left, bk_right, background = [],[],[]
	y_bk_left, y_bk_right = [], []
	
        binSize = config.DATABASE["binSize"] 
        ext = config.DATABASE["extend_context"]
	window = int(binSize/2) # used for the background sample data

	if dataAug > 0:
            bk_left_aug, bk_right_aug, background_aug = [],[],[]
            y_bk_left_aug, y_bk_right_aug = [],[]

        # interate over all regions
	for i in range(len(rgList)):
            
            xlen = rgList[i][2] - rgList[i][1]
            cnvType = rgList[i][3]

            ##1. break point sample, from (1 to binSize-1) not fix to the center part
            ## in the current implementation, the right and right shift is the same
            shift = random.randint(1, binSize-1)
            bk_left.append((rgList[i][0], rgList[i][1]-shift-ext, rgList[i][1]+binSize-shift+ext, cnvType, "L", shift+ext ,rgList[i][4], rgList[i][5]))
	    bk_right.append((rgList[i][0], rgList[i][2]-shift-ext, rgList[i][2]+binSize-shift+ext, cnvType, "R", shift+ext ,rgList[i][6], rgList[i][7]))
	    
            # do re-checking for this function
	    y_left, y_right = getPairMask(cnvType, rgList[i], shift, binSize)
	    y_bk_left.append(y_left)
            y_bk_right.append(y_right)
		
            ## do more than one time of shifting center postions
            if dataAug > 0:
                for rep in range(dataAug):
                    shift = random.randint(1, binSize-1)
                    bk_left_aug.append((rgList[i][0], rgList[i][1]-shift-ext, rgList[i][1]+binSize-shift+ext, cnvType, "L", shift+ext, rgList[i][4], rgList[i][5]))
                    bk_right_aug.append((rgList[i][0], rgList[i][2]-shift-ext, rgList[i][2]+binSize-shift+ext, cnvType, "R", shift+ext, rgList[i][6], rgList[i][7]))

		    y_left, y_right = getPairMask(cnvType, rgList[i], shift, binSize)
                    y_bk_left_aug.append(y_left)
                    y_bk_right_aug.append(y_right)

            ##2. Negative background sample, based on the large distance random shifting and break point checking. 
            while(True):
                shift = random.randint(binSize*2, binSize*100)
                if rgList[i][1] -shift-window-ext < 0: continue
                rg_cand = (rgList[i][0], rgList[i][1]-shift-window-ext, rgList[i][1]-shift+window+ext, "BG", "L", -1, 1, -1)
                if whetherNotBKregion(rg_cand, bkDic) == True:
                    background.append(rg_cand)
                    break

            while(True):
		shift = random.randint(binSize*2, binSize*100)
                rg_cand = (rgList[i][0], rgList[i][2]+shift-window-ext, rgList[i][2]+shift+window+ext, "BG", "R", -1,  1, -1)
                if whetherNotBKregion(rg_cand, bkDic) == True:
                    background.append(rg_cand)
                    break

            ###20190110 @@ revious for testing the data augmentation issue
            if dataAug > 0:
                for rep in range(dataAug):
                    while(True):
		        shift = random.randint(binSize*2, binSize*100)
		        local_window = random.randint(1, binSize-1)
                        if rgList[i][1] -shift -local_window-ext < 0: continue
                        rg_cand = (rgList[i][0], rgList[i][1]-shift-local_window-ext, rgList[i][1]-shift+binSize-local_window+ext, "BG", "L", -1, 1, -1)
                        if whetherNotBKregion(rg_cand, bkDic) == True:
                            background_aug.append(rg_cand)
                            break

                    while(True):
		        shift = random.randint(binSize*2, binSize*100)
		        local_window = random.randint(1, binSize-1)
                        rg_cand = (rgList[i][0], rgList[i][2]+shift-local_window-ext, rgList[i][2]+shift+binSize -local_window+ext, "BG", "R", -1,  1, -1)
                        if whetherNotBKregion(rg_cand, bkDic) == True:
                            background_aug.append(rg_cand)
                            break
        

        logger.info("** Break point and background data has been generated completely !!!")
        
	## retrive related information from Bam file, based on a region range.
	rd_bl, seq_bl, ma_bl, gc_bl = input_gen_fromRgs(bk_left, bamFile)
	rd_br, seq_br, ma_br, gc_br = input_gen_fromRgs(bk_right, bamFile)
        if bk_in: rd_bg, seq_bg, ma_bg, gc_bg = input_gen_fromRgs(background, bamFile)

        # checking step 
        ind = np.apply_along_axis(check_all_zero, 1, rd_bl)
        logger.debug("\n** checking all zero for bk-left %d" %(np.sum(np.sum(ind))))
        ind = np.apply_along_axis(check_all_zero, 1, rd_br)
        logger.debug("** checking all zero for bk-right %d" %(np.sum(np.sum(ind))))
        if bk_in:
            ind = np.apply_along_axis(check_all_zero, 1, rd_bg)
            logger.debug("** checking all zero for background %d" %(np.sum(np.sum(ind))))

        if dataAug > 0:
            rd_bl_aug, seq_bl_aug, ma_bl_aug, gc_bl_aug = input_gen_fromRgs(bk_left_aug, bamFile)
	    rd_br_aug, seq_br_aug, ma_br_aug, gc_br_aug = input_gen_fromRgs(bk_right_aug, bamFile)
            
            ### 20190110 @@ revious for testing the data augmentation issue
            if bk_in: rd_bg_aug, seq_bg_aug, ma_bg_aug, gc_bg_aug = input_gen_fromRgs(background_aug, bamFile)
	

            ind = np.apply_along_axis(check_all_zero, 1, rd_bl_aug)
            logger.debug("** AUG-checking all zero for bk-left %d" %(np.sum(np.sum(ind))))
            ind = np.apply_along_axis(check_all_zero, 1, rd_br_aug)
            logger.debug("** AUG-checking all zero for bk-right %d" %(np.sum(np.sum(ind))))
            
            ### 20190110 @@ revious for testing the data augmentation issue
            if bk_in:
                ind = np.apply_along_axis(check_all_zero, 1, rd_bg_aug)
                logger.debug("** AUG-checking all zero for background %d" %(np.sum(np.sum(ind))))
            

	y_bg = [[0]*binSize]*len(background)


	x_data, y_data, seq_data, rgs_data, gc_data = [], [], [], [], []
        ###################################################
        # 1. basic data without data augmentation
        ###################################################
        x_data.extend(rd_bl)
	x_data.extend(rd_br)
        if bk_in: x_data.extend(rd_bg)
		
	y_data.extend(y_bk_left)
	y_data.extend(y_bk_right)
        if bk_in: y_data.extend(y_bg)
		
	seq_data.extend(seq_bl)
	seq_data.extend(seq_br)
        if bk_in: seq_data.extend(seq_bg)
	        
	rgs_data.extend(bk_left)
        rgs_data.extend(bk_right)
        if bk_in: rgs_data.extend(background)
	
	gc_data.extend(gc_bl)
	gc_data.extend(gc_br)
        if bk_in: gc_data.extend(gc_bg)

        ## The augmenation code is not checked!!!!!!
        if dataAug > 0:

            x_data.extend(rd_bl_aug)
            y_data.extend(y_bk_left_aug)
            seq_data.extend(seq_bl_aug)
            rgs_data.extend(bk_left_aug)
            gc_data.extend(gc_bl_aug)

            x_data.extend(rd_br_aug)
            y_data.extend(y_bk_right_aug)
            seq_data.extend(seq_br_aug)
            rgs_data.extend(bk_right_aug)
            gc_data.extend(gc_br_aug)

            
            ###20199110  @@ revious for testing the data augmentation issue
        
            if bk_in:
                x_data.extend(rd_bg_aug)
                y_data.extend([ [0]* binSize ]*rd_bg_aug.shape[0])
                seq_data.extend(seq_bg_aug)
                rgs_data.extend(background_aug)
                gc_data.extend(gc_bg_aug)

        ## do the shuffling for the data
        idx = range(len(y_data))
        random.shuffle(idx)
	
        x_data = [ x_data[i] for i in idx ]
        y_data = [ y_data[i] for i in idx ]
        seq_data = [ seq_data[i] for i in idx ]
	rgs_data = [ rgs_data[i] for i in idx ]
        gc_data = [ gc_data[i] for i in idx ]

        # do not return too early which will reduce the effect of data augmentation.
        return {"x":x_data, "y":y_data, "seq":seq_data, "rgs":rgs_data, "gc":gc_data}
	

"""
Prepareing the training data for BK prediction models
Consider remove the sequence generation part in the final version, as the sequences shows no improvment for the final performance. 
"""

def bkDataGen_crossSample(goldFile, bamFile, dataAug=0):

        random.seed(config.DATABASE["rand_seed"])
	binSize = config.DATABASE["binSize"]

        rgs_train = load_region_file(goldFile[0])
        rgs_train = [rg for rg in rgs_train if (rg[0] != "x" and rg[0] != "y")]
        bkDic = genBkDic(rgs_train)
        rgs_train_short50 = [rg for rg in rgs_train if rg[2] - rg[1] < 50 ]

        rgs_test = load_region_file(goldFile[1])
        rgs_test = [rg for rg in rgs_test if (rg[0] != "x" and rg[0] != "y")]
        rgs_test_short50 = [rg for rg in rgs_test if rg[2] - rg[1] < 50 ]
        

        print("\n********* [Cross sample] BK regions **********")
        print("\t* Training sample %s has [%d] regions" %(goldFile[0], len(rgs_train)))
        print("\t @@ Note number of CNV regions <=50bp: %d" %(len(rgs_train_short50)))
        print("\t* Testing sample %s has [%d] regions" %(goldFile[1], len(rgs_test)))
        print("\t @@ Note number of CNV regions <=50bp: %d" %(len(rgs_test_short50)))

        train_data = genData_from_rgLists(rgs_train, bkDic, bamFile[0], dataAug, bk_in=True)
        test_data =  genData_from_rgLists(rgs_test, bkDic, bamFile[1], dataAug=0,bk_in=True)
        
        return (train_data, test_data)


def bkDataGen_singleSample(goldFile, bamFile, dataAug=0, splitFold=0.2):
	
        # fix the random seed for experiment comparision
        random.seed(config.DATABASE["rand_seed"])
	binSize = config.DATABASE["binSize"]

        # loading regions and filtering 
        rgs = load_region_file(goldFile[0])
        # autosome
        rgs = [rg for rg in rgs if (rg[0] != "x" and rg[0] != "y")]
        bkDic = genBkDic(rgs)


        rgs_del_all = [rg for rg in rgs if rg[3].__contains__("DEL")] 
        rgs_dup = [rg for rg in rgs if rg[3]== "DUP"]
        rgs_cnv = [rg for rg in rgs if rg[3]== "CNV"]


        rgs_del = [rg for rg in rgs if rg[3] == "DEL"] 
        rgs_del_line1 = [rg for rg in rgs if rg[3] == "DEL_LINE1"]
        rgs_del_sva = [rg for rg in rgs if rg[3] == "DEL_SVA"]
        rgs_del_alu = [rg for rg in rgs if rg[3] == "DEL_ALU"]
        
        print("\n********* [Single sample] BK regions **********")
	print("* Total number of GS region is [%d]" %(len(rgs)))
        print("\t** TYPE=DUP, number=%d" %(len(rgs_dup)))
        print("\t** TYPE=CNV, number=%d" %(len(rgs_cnv)))

        print("\t** TYPE=DEL(all), number=%d" %(len(rgs_del_all)))
        print("\t*** TYPE=DEL, number=%d" %(len(rgs_del)))
        print("\t*** TYPE=DEL_LINE1, number=%d" %(len(rgs_del_line1)))
        print("\t*** TYPE=DEL_ALU, number=%d" %(len(rgs_del_alu)))
        print("\t*** TYPE=DEL_SVA, number=%d" %(len(rgs_del_sva)))

        rgs_short50 = [rg for rg in rgs if rg[2]-rg[1] < 50 ]
        print("\t@@ Note number of CNV regions <=50bp: %d" %(len(rgs_short50)))
        

        ##############################################################
        # generating data according to the different model
        #############################################################
        if config.DATABASE["data_split"] == "CV":
            train_data= genData_from_rgLists(rgs, bkDic, bamFile[0], dataAug=0, bk_in=True)
            
            # position taking, not actually used, no use
            test_data =  genData_from_rgLists(rgs_del, bkDic, bamFile[0], dataAug=0,bk_in=True)
            return (train_data, test_data)

        ################################################################
        # if not testing CV, first, split the data to 20% , then evluate the augmented and non-augmented results
        ################################################################
        # """ this part is used for the evaluation of data Augmentation. 
        if config.DATABASE["data_split"] == "RandRgs":
  
            idx = range(len(rgs))
            random.shuffle(idx)
            sidx = int(splitFold * len(rgs))
            rgList = [ rgs[i] for i in idx ]
            
            rgList_test = rgList[:sidx]
            rgList_train = rgList[sidx:]
            
            test_data =  genData_from_rgLists(rgList_test,  bkDic, bamFile[0], dataAug=0, bk_in=True)
            train_data = genData_from_rgLists(rgList_train, bkDic, bamFile[0], dataAug,   bk_in=True)
            
            return (train_data, test_data)

        if config.DATABASE["data_split"] == "Stratify":

            rgList_test = []
            rgList_train = []

            # DEL
            idx = range(len(rgs_del))
            random.shuffle(idx)
            sidx = int(splitFold * len(rgs_del))
            rgs_del = [ rgs_del[i] for i in idx ]
    
            rgList_test.extend(rgs_del[:sidx])
            rgList_train.extend(rgs_del[sidx:])

            ##DEL_LINE1
            idx = range(len(rgs_del_line1))
            random.shuffle(idx)
            sidx = int(splitFold * len(rgs_del_line1))
            rgs_del_line1 = [ rgs_del_line1[i] for i in idx ]

            rgList_test.extend(rgs_del_line1[:sidx])
            rgList_train.extend(rgs_del_line1[sidx:])

            ##DLE_ALU
            idx = range(len(rgs_del_alu))
            random.shuffle(idx)
            sidx = int(splitFold * len(rgs_del_alu))
            rgs_del_alu = [ rgs_del_alu[i] for i in idx ]
    
            rgList_test.extend(rgs_del_alu[:sidx])
            rgList_train.extend(rgs_del_alu[sidx:])

            ##
            idx = range(len(rgs_del_sva))
            random.shuffle(idx)
            sidx = int(splitFold * len(rgs_del_sva))
            rgs_del_sva = [ rgs_del_sva[i] for i in idx ]
    
            rgList_test.extend(rgs_del_sva[:sidx])
            rgList_train.extend(rgs_del_sva[sidx:])


            # DUP
            idx = range(len(rgs_dup))
            random.shuffle(idx)
            sidx = int(splitFold * len(rgs_dup))
            rgs_dup = [ rgs_dup[i] for i in idx ]

            rgList_test.extend(rgs_dup[:sidx])
            rgList_train.extend(rgs_dup[sidx:])


            # CNV
            idx = range(len(rgs_cnv))
            random.shuffle(idx)
            sidx = int(splitFold * len(rgs_cnv))
            rgs_cnv = [ rgs_cnv[i] for i in idx ]

            rgList_test.extend(rgs_cnv[:sidx])
            rgList_train.extend(rgs_cnv[sidx:])

  
            # random shuffle again
            idx = range(len(rgList_test))
            random.shuffle(idx)
            rgList_test = [ rgList_test[i] for i in idx ]

            idx = range(len(rgList_train))
            random.shuffle(idx)
            rgList_train = [ rgList_train[i] for i in idx ]

            test_data =  genData_from_rgLists(rgList_test,  bkDic, bamFile[0], dataAug=0, bk_in=True)
            train_data = genData_from_rgLists(rgList_train, bkDic, bamFile[0], dataAug,   bk_in=True)


            return (train_data, test_data)


        # make sure the test contains DUP
        if config.DATABASE["data_split"] == "Stratify":

            rgList_test = []
            rgList_train = []

            # DEL
            idx = range(len(rgs_del))
            random.shuffle(idx)
            sidx = int(splitFold * len(rgs_del))
            rgs_del = [ rgs_del[i] for i in idx ]
            
            rgList_test.extend(rgs_del[:sidx])
            rgList_train.extend(rgs_del[sidx:])

            # DUP
            idx = range(len(rgs_dup))
            random.shuffle(idx)
            sidx = int(splitFold * len(rgs_dup))
            rgs_dup = [ rgs_dup[i] for i in idx ]

            rgList_test.extend(rgs_dup[:sidx])
            rgList_train.extend(rgs_dup[sidx:])

            # CNV
            idx = range(len(rgs_cnv))
            random.shuffle(idx)
            sidx = int(splitFold * len(rgs_cnv))
            rgs_cnv = [ rgs_cnv[i] for i in idx ]

            rgList_test.extend(rgs_cnv[:sidx])
            rgList_train.extend(rgs_cnv[sidx:])
  
            # random shuffle again
            idx = range(len(rgList_test))
            random.shuffle(idx)
            rgList_test = [ rgList_test[i] for i in idx ]

            idx = range(len(rgList_train))
            random.shuffle(idx)
            rgList_train = [ rgList_train[i] for i in idx ]

            test_data =  genData_from_rgLists(rgList_test,  bkDic, bamFile[0], dataAug=0, bk_in=True)
            train_data = genData_from_rgLists(rgList_train, bkDic, bamFile[0], dataAug,   bk_in=True)


            return (train_data, test_data)
            

        #############################################################
        # DEL training and testing on DUP
        #############################################################
        if config.DATABASE["data_split"] == "DEL-DUP":
            
            test_data =  genData_from_rgLists(rgs_dup, bkDic, bamFile[0], dataAug=0,bk_in=True)
            train_data = genData_from_rgLists(rgs_del, bkDic, bamFile[0], dataAug, bk_in=True)
            
            return (train_data, test_data)

	

#################################################################################
## Caching related data
#################################################################################

def cache_trainData(goldFile, bamFile, dataPath, dataSplit):

    print("\n[Bin Sample]: >> generating bin Sample data ... ")

    # first saving all the available data into caching. then split data later
    
    # this is the place catching the data. current is directly save the data

    
    if config.DATABASE["eval_mode"] == "single":
        cv_data, cnv_test_data = bkDataGen_singleSample(goldFile, bamFile, config.DATABASE["data_aug"])
    
    # this part need to be carefully treated
    if config.DATABASE["eval_mode"] == "cross":
        cv_data, cnv_test_data = bkDataGen_crossSample(goldFile, bamFile, config.DATABASE["data_aug"])


    with h5py.File(dataPath, 'w') as hf:
	
	# CV data set
        hf.create_dataset("x_data", data=cv_data["x"])
        hf.create_dataset("y_data", data=cv_data["y"])
	hf.create_dataset("seq_data", data=cv_data["seq"])
	hf.create_dataset("rgs_data", data=cv_data["rgs"])
	hf.create_dataset("gc_data", data=cv_data["gc"])

        # test
        hf.create_dataset("x_test",   data=cnv_test_data["x"])
        hf.create_dataset("y_test",   data=cnv_test_data["y"])
        hf.create_dataset("seq_test", data=cnv_test_data["seq"])
        hf.create_dataset("rgs_test", data=cnv_test_data["rgs"])
        hf.create_dataset("gc_test", data=cnv_test_data["gc"])

def load_cache_trainData(dataPath):

    with h5py.File(dataPath,'r') as hf:
        x_data = hf["x_data"][:]
        y_data = hf["y_data"][:]
        seq_data = hf["seq_data"][:]
        rgs_data = hf["rgs_data"][:]
        gc_data = hf["gc_data"][:]

        x_test = hf["x_test"][:]
        y_test = hf["y_test"][:]
        seq_test = hf["seq_test"][:]
        rgs_test = hf["rgs_test"][:]
        gc_test = hf["gc_test"][:]

    return (x_data, y_data, seq_data, rgs_data, gc_data, \
            x_test, y_test, seq_test, rgs_test, gc_test)



"""
if __name__ == "__main__":
    
    goldFile = "../data/CNV_annotation/Haraksingh/NA12878_1000GP_gold.txt"
    lowBamFile="/home/dev/NA12878.mapped.ILLUMINA.bwa.CEU.low_coverage.20121211.bam"
    #highBamFile="/home/dev/NA12878.mapped.ILLUMINA.bwa.CEU.high_coverage_pcr_free.20130906.rmdup.bam"
    # uhighBamFile="/data/Dataset/GIAB/JP201712_data/HG001_NA12878/HG001.hs37d5.300x.bam"

    bkDataGen_withSeq(goldFile, lowBamFile, 0, autosomeOnly=True)
"""
