# config file
## Note: some option values can be modified through the command option.
## Make proper setting before running the code. 


DATAPATH="../data/"
BAMFOLD="/data/Dataset/1000GP/phase3/"

DATABASE={	
    # reference genome sequence
    'ref_faFile': DATAPATH + "reference/hg19/hs37d5.fa",
    'chr_prefix':False,

    # reference general chromesome information
    'chrLen_info': DATAPATH + "reference/hg19/hg19_chr_info.txt",
    # Read signal types
    'count_type':"RD",
    # bin Size of the the data
    'binSize':400,
    # mappability files
    'mappability_file': DATAPATH  + "reference/hg19/wgEncodeCrgMapabilityAlign100mer.bigWig",
    'mappability_threshold':0.9,
    # alignment threshold for filtering out low quality data
    'mapq_threshold':30, 
    'base_quality_threshold':30,
    # regions avoid in the reference genome
    'black_list':"../data/reference/hg19/hg19.blackList.bed",
    # Embedding related
    'kmer_dic':None,
    'max_kmer':1,
    #random seed
    'rand_seed':1234,

    # background genome sample rate
    'genomeSampleRate':0.01,
    # data augmentation option
    'data_aug':0,

    'extend_context':0,
    'extra_feat_ext_len':1,

    # different data running model: CV (all bks data), DEL-DUP, rSplit
    'data_split':"RandRgs",

    # evluation model:  "single" or "cross"
    'eval_mode':"single",
    'model_param': "",

    'vcf': "",
    'vcf2': "",
    "vcf_filter":0,
    "vcf_filter2":1,
    "vcf_ci":99999999,
    "small_cv_train":"normal",

    "fix_center":False,
    "shift_low_bound":10,

    'USESEQ':False,
    'USEPROB':False,
    'GC_NORM':False,

    # output folds
    ## fold for saving the training SV regions
    'outFold_train_rgs':"../experiment/train_rgs/",
    ## fold for saving the enhancement results
    "enhance_output_fold":"../experiment/enhanced_result/",
    ## fold for saving breakpoint change matrix, plotted in heatmap
    "heatmap_fold":"../experiment/heatmap/",
    "model_data_tag":"",

    # threshold of overlapped (Jaccard similarity)
    "JS":0.5, 
}


"""
BAMFILE={
        # NA12878
        "na12878_7x":  BAMFOLD+"NA12878/NA12878.mapped.ILLUMINA.bwa.CEU.low_coverage.20121211.bam",
        "na12878_60x": BAMFOLD+"NA12878/NA12878.mapped.ILLUMINA.bwa.CEU.high_coverage_pcr_free.20130906.bam",

        "na12878_60x_ds0.1": BAMFOLD+"NA12878/NA12878.mapped.ILLUMINA.bwa.CEU.high_coverage_pcr_free.20130906.downsample0.1.bam",
        "na12878_60x_ds0.2": BAMFOLD+"NA12878/NA12878.mapped.ILLUMINA.bwa.CEU.high_coverage_pcr_free.20130906.downsample0.2.bam",
        "na12878_60x_ds0.3": BAMFOLD+"NA12878/NA12878.mapped.ILLUMINA.bwa.CEU.high_coverage_pcr_free.20130906.downsample0.3.bam",
        "na12878_60x_ds0.4": BAMFOLD+"NA12878/NA12878.mapped.ILLUMINA.bwa.CEU.high_coverage_pcr_free.20130906.downsample0.4.bam",
        "na12878_60x_ds0.5": BAMFOLD+"NA12878/NA12878.mapped.ILLUMINA.bwa.CEU.high_coverage_pcr_free.20130906.downsample0.5.bam",
        "na12878_60x_ds0.7": BAMFOLD+"NA12878/NA12878.mapped.ILLUMINA.bwa.CEU.high_coverage_pcr_free.20130906.downsample0.7.bam",

        # NA19238
        "na19238_7x":BAMFOLD+"NA19238.mapped.ILLUMINA.bwa.YRI.low_coverage.20130415.bam",
        "na19238_60x":BAMFOLD+"NA19238.mapped.ILLUMINA.bwa.YRI.high_coverage_pcr_free.20130924.bam",

        # NA18939
        "na18939_7x":BAMFOLD+"NA18939.mapped.ILLUMINA.bwa.JPT.low_coverage.20130415.bam",
        "na18939_60x":BAMFOLD+"NA18939.wgs.ILLUMINA.bwa.JPT.high_cov_pcr_free.20140203.bam",
}
"""
"""
ic=-1
AnnoCNVFile ={       
        "na12878_7x": "../data/SV_annotation/1KGP/annoFile/NA12878_1kGP_IC-"+str(ic)+".bed",
        "na12878_60x": "../data/SV_annotation/1KGP/annoFile/NA12878_1kGP_IC-"+str(ic)+".bed",

        "na12878_60x_ds0.1": "../data/SV_annotation/1KGP/annoFile/NA12878_1kGP_IC-"+str(ic)+".bed",
        "na12878_60x_ds0.2": "../data/SV_annotation/1KGP/annoFile/NA12878_1kGP_IC-"+str(ic)+".bed",
        "na12878_60x_ds0.3": "../data/SV_annotation/1KGP/annoFile/NA12878_1kGP_IC-"+str(ic)+".bed",
        "na12878_60x_ds0.4": "../data/SV_annotation/1KGP/annoFile/NA12878_1kGP_IC-"+str(ic)+".bed",
        "na12878_60x_ds0.5": "../data/SV_annotation/1KGP/annoFile/NA12878_1kGP_IC-"+str(ic)+".bed",
        "na12878_60x_ds0.7": "../data/SV_annotation/1KGP/annoFile/NA12878_1kGP_IC-"+str(ic)+".bed",

        "na19238_7x": "../data/SV_annotation/1KGP/annoFile/nonOverlap_NA19238_1kGP_IC-"+str(ic)+".bed",
        "na19238_60x": "../data/SV_annotation/1KGP/annoFile/nonOverlap_NA19238_1kGP_IC-"+str(ic)+".bed",
        
        "na19239_7x": "../data/SV_annotation/1KGP/annoFile/nonOverlap_NA19239_1kGP_IC-"+str(ic)+".bed",
        "na19239_60x": "../data/SV_annotation/1KGP/annoFile/nonOverlap_NA19239_1kGP_IC-"+str(ic)+".bed",            
}
"""

