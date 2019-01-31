# config file
## Note: some option values can be modified through the command option.

DATABASE ={	
    # reference genome sequence
    'ref_faFile':"../data/reference/hs37d5.fa",
    # reference general chromesome information
    'chrLen_info':"../data/reference/hg19_chr_info.txt",
	# Read signal types
	'count_type':"RD",
    # bin Size of the the data
    'binSize':1000,
    # mappability files
    'mappability_file':"../data/reference/wgEncodeCrgMapabilityAlign100mer.bigWig",
    'mappability_threshold':0.9,
    # alignment threshold for filtering out low quality data
    'mapq_threshold':30, 
    'base_quality_threshold':30,
    # regions avoid in the reference genome
	'black_list':"../data/reference/hg19.blackList.bed",
    # Embedding related
	'kmer_dic':None,
	'max_kmer':1,
    # random seed
    'rand_seed':1234,
    # background genome sample rate
    'genomeSampleRate':0.01,
    # data augmentation option
    'data_aug':0,
    'extend_context':0,
    # different data running model: CV (all bks data), DEL-DUP, rSplit
    'data_split':"RandRgs",
    # evluation model:  "single" or "cross"
    'eval_mode':"single",
    'model_param': ""
}

BAMFOLD="/data/Dataset/1000GP/phase3/"

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

        "na12878_300x": "/data/Dataset/GIAB/JP201712_data/HG001_NA12878/HG001.hs37d5.300x.bam",
        "na12878_pacbio": "/data2/Dataset/PacBio/NA12878/sorted_final_merged.bam",

        # NA19238
        "na19238_7x":"/data2/Dataset/1000GP/phase3/NA19238/NA19238.mapped.ILLUMINA.bwa.YRI.low_coverage.20130415.bam",
        "na19238_60x":"/data2/Dataset/1000GP/phase3/NA19238/NA19238.mapped.ILLUMINA.bwa.YRI.high_coverage_pcr_free.20130924.bam",

        # NA12239
        "na19239_7x":"/data2/Dataset/1000GP/phase3/NA19239/NA19239.mapped.ILLUMINA.bwa.YRI.low_coverage.20130415.bam",
        "na19239_60x":"/data2/Dataset/1000GP/phase3/NA19239/NA19239.mapped.ILLUMINA.bwa.YRI.high_coverage_pcr_free.20130924.bam",

        # HG00419
        "hg00419_7x":"/data2/Dataset/1000GP/phase3/HG00419/HG00419.mapped.ILLUMINA.bwa.CHS.low_coverage.20130415.bam",
        "hg00419_60x":"/data2/Dataset/1000GP/phase3/HG00419/HG00419.wgs.ILLUMINA.bwa.CHS.high_cov_pcr_free.20140203.bam",

        # NA18939
        "na18939_7x":"/data2/Dataset/1000GP/phase3/NA18939/NA18939.mapped.ILLUMINA.bwa.JPT.low_coverage.20130415.bam",
        "na18939_60x":"/data2/Dataset/1000GP/phase3/NA18939/NA18939.wgs.ILLUMINA.bwa.JPT.high_cov_pcr_free.20140203.bam",

}

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

        "na12878_300x": "../data/SV_annotation/1KGP/annoFile/NA12878_1kGP_IC-"+str(ic)+".bed",


        "na19238_7x": "../data/SV_annotation/1KGP/annoFile/nonOverlap_NA19238_1kGP_IC-"+str(ic)+".bed",
        "na19238_60x": "../data/SV_annotation/1KGP/annoFile/nonOverlap_NA19238_1kGP_IC-"+str(ic)+".bed",
        
        "na19239_7x": "../data/SV_annotation/1KGP/annoFile/nonOverlap_NA19239_1kGP_IC-"+str(ic)+".bed",
        "na19239_60x": "../data/SV_annotation/1KGP/annoFile/nonOverlap_NA19239_1kGP_IC-"+str(ic)+".bed",


        "hg00419_7x": "../data/SV_annotation/1KGP/annoFile/nonOverlap_HG00419_1kGP_IC-"+str(ic)+".bed",
        "hg00419_60x": "../data/SV_annotation/1KGP/annoFile/nonOverlap_HG00419_1kGP_IC-"+str(ic)+".bed",

        "na18939_7x":"../data/SV_annotation/1KGP/annoFile/nonOverlap_NA18939_1kGP_IC-"+str(ic)+".bed",
        "na18939_60x":"../data/SV_annotation/1KGP/annoFile/nonOverlap_NA18939_1kGP_IC-"+str(ic)+".bed",
        
}

