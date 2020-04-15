# uncompyle6 version 3.2.4
# Python bytecode 2.7 (62211)
# Decompiled from: Python 2.7.6 (default, Nov 23 2017, 15:49:48) 
# [GCC 4.8.4]
# Embedded file name: /data/workspace/deepCNV_1.0/code/data_bam2rd.py
# Compiled at: 2018-11-07 20:07:17
"""
Data:   2018-05-29
Author: Yao-zhong Zhang @ IMSUT

Description:
Functions of reading bam file and convert to readCount/readDepth signals.

1. Definition
readDepth = number of pileup reads for the coordinate.
readCount_bin = number of reads that started in the bin.  

2018-08-15 update the RD count with mapq filters for each position
"""
from util import *
#import numpy as np
DTYPE = np.int32

def getRegion_RC_worker(bamfileName, region, binSize):

    bamfile = pysam.AlignmentFile(bamfileName, 'rb')
    nSegs = (region[2] - region[1]) / binSize
    region_rd = np.zeros(nSegs, dtype=DTYPE)
    if region[1] + nSegs * binSize < region[2]:
        end_POS = region[1] + nSegs * binSize
    else:
        end_POS = region[2]
    reads = bamfile.fetch(region[0], region[1], end_POS)
    for r in reads:
        if r.pos < r.mpos and r.pos >= region[1] and r.pos < end_POS:
            pos_rel = (r.pos - region[1]) / binSize
            if r.mapping_quality >= config.DATABASE['mapq_threshold']:
                region_rd[pos_rel] += 1

    bamfile.close()
    return region_rd


def getRegion_RC_list_worker(param):
    bamfileName, regions = param
    with pysam.AlignmentFile(bamfileName, 'rb') as (bamfile):
        region_rd = np.zeros(len(regions), dtype=DTYPE)
        for i, region in enumerate(regions):
            reads = bamfile.fetch(region[0], region[1], region[2])
            for r in reads:
                if r.pos < r.mpos and r.pos >= region[1] and r.pos < region[2]:
                    if r.mapping_quality >= config.DATABASE['mapq_threshold']:
                        region_rd[i] += 1

    return region_rd


def getRegion_RCVec_list_worker(param):
    bamfileName, regions = param

    with pysam.AlignmentFile(bamfileName, 'rb') as (bamfile):
        regions_rcMat = np.zeros((len(regions), config.DATABASE['binSize']), dtype=DTYPE)
        region_rd = np.zeros(len(regions), dtype=DTYPE)
        for i, region in enumerate(regions):
            reads = bamfile.fetch(region[0], region[1], region[2])
            for r in reads:
                if r.pos < r.mpos and r.pos >= region[1] and r.pos < region[2]:
                    if r.mapping_quality >= config.DATABASE['mapq_threshold']:
                        regions_rcMat[(i, r.pos - region[1])] += 1

    return regions_rcMat

# workers for exatraction input signals for the target regions of assigned
def getRegion_RDVec_list_worker(param):  
    bamfileName, regions = param
    # check pysam alignement filges
    with pysam.AlignmentFile(bamfileName, 'rb') as (bamfile):
        regions_rdMat = np.zeros((len(regions), config.DATABASE['binSize']), dtype=DTYPE)
        for i, region in enumerate(regions):

            pileup = bamfile.pileup(region[0], region[1], region[2])
            for pColumn in pileup:
                if pColumn.pos >= region[1] and pColumn.pos < region[2]:
                    pColumn.set_min_base_quality(config.DATABASE['base_quality_threshold'])
                    regions_rdMat[(i, pColumn.pos - region[1])] = pColumn.get_num_aligned()

    return regions_rdMat


def getRegion_RD_list_worker(param):
    bamfileName, regions = param
    with pysam.AlignmentFile(bamfileName, 'rb') as (bamfile):
        region_rd = np.zeros(len(regions), dtype=DTYPE)
        for i, region in enumerate(regions):
            pileup = bamfile.pileup(region[0], region[1], region[2])
            for pColumn in pileup:
                if pColumn.pos >= region[1] and pColumn.pos < region[2]:
                    pColumn.set_min_base_quality(config.DATABASE['mapq_threshold'])
                    region_rd[i] += pColumn.get_num_aligned()

    return region_rd


def getRegion_RC_pool(param):
    bamfileName, region, binSize = param
    return getRegion_RC_worker(bamfileName, region, binSize)


def getRegion_RC_worker_sharedMem_return(bamfileName, region, binSize, output, c):
    bamfile = pysam.AlignmentFile(bamfileName, 'rb')
    nSegs = (region[2] - region[1]) / binSize
    region_rd = np.zeros(nSegs, dtype=DTYPE)
    if region[1] + nSegs * binSize < region[2]:
        end_POS = region[1] + nSegs * binSize
    else:
        end_POS = region[2]
    reads = bamfile.fetch(region[0], region[1], end_POS)
    for r in reads:
        if r.pos < r.mpos and r.pos >= region[1] and r.pos < end_POS:
            pos_rel = (r.pos - region[1]) / binSize
            region_rd[pos_rel] += 1

    output[c] = region_rd
    bamfile.close()


def get_RC_parallel(bamfileName, chr_rg, binSize=500):
    cores = multiprocessing.cpu_count()
    logger.debug('* Total [%d]-cores are used for processing ...', cores)
    nSegs = (chr_rg[2] - chr_rg[1]) / binSize
    nThreadUnit = nSegs / cores
    process = []
    output = multiprocessing.Manager().dict()
    for c in range(cores):
        start_local = c * nThreadUnit * binSize
        end_local = (c + 1) * nThreadUnit * binSize
        rg = (chr_rg[0], start_local, end_local)
        p = multiprocessing.Process(target=getRegion_RC_worker_sharedMem_return, args=(bamfileName, rg, binSize, output, c))
        p.daemon = True
        p.start()
        process.append(p)

    end_position = nThreadUnit * cores * binSize
    if chr_rg[2] > end_position:
        rg = (
         chr_rg[0], end_position, chr_rg[2])
        p = multiprocessing.Process(target=getRegion_RC_worker_sharedMem_return, args=(bamfileName, rg, binSize, output, cores))
        p.daemon = True
        p.start()
        process.append(p)
    for p in process:
        p.join()

    rc_bin = np.concatenate([ output[c] for c in range(len(process)) ])
    return rc_bin

# currently used for extrction
def get_RDVec_parallel_nobin(bamfileName, rgs):

    cores = multiprocessing.cpu_count()
    
    if len(rgs) < cores:
        cores = len(rgs)

    pool = multiprocessing.Pool(processes=cores)
    step = int(len(rgs) / cores)

    rgsList = [ rgs[i * step:(i + 1) * step] for i in range(cores - 1) ]
    
    rgsList.append(rgs[(cores - 1) * step:])
    todo_params = [ (bamfileName, rs) for rs in rgsList ]
    output = pool.map(getRegion_RDVec_list_worker, todo_params)

    pool.close()
    pool.join()
    vec = np.concatenate(output)
    return vec


def get_RC_parallel_nobin(bamfileName, rgs):
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    step = int(len(rgs) / cores)
    rgsList = [ rgs[i * step:(i + 1) * step] for i in range(cores - 1) ]
    rgsList.append(rgs[(cores - 1) * step:])
    todo_params = [ (bamfileName, rs) for rs in rgsList ]
    output = pool.map(getRegion_RC_list_worker, todo_params)
    pool.close()
    pool.join()
    vec = np.concatenate(output)
    return vec


def getChr_RC_parallel_pool(bamfileName, chr_rg, binSize=500):
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    nSegs = (chr_rg[2] - chr_rg[1]) / binSize
    nThreadUnit = nSegs / cores
    todo_params = []
    for c in range(cores):
        start_local = c * nThreadUnit * binSize
        end_local = (c + 1) * nThreadUnit * binSize
        rg = (chr_rg[0], start_local, end_local)
        todo_params.append((bamfileName, rg, binSize))

    end_position = nThreadUnit * cores * binSize
    if chr_rg[2] > end_position:
        rg = (
         chr_rg[0], end_position, chr_rg[2])
        todo_params.append((bamfileName, rg, binSize))
    output = pool.map(getRegion_RC_pool, todo_params)
    pool.close()
    pool.join()
    vec = np.concatenate(output)
    return vec


def getRegion_RDvec(bamfile, region):
    rlen = region[2] - region[1]
    region_rd = np.zeros(rlen, dtype=DTYPE)
    pileup = bamfile.pileup(region[0], region[1], region[2])
    for pColumn in pileup:
        if pColumn.pos >= region[1] and pColumn.pos < region[2]:
            pColumn.set_min_base_quality(config.DATABASE['mapq_threshold'])
            region_rd[pColumn.pos - region[1]] = pColumn.get_num_aligned()

    return region_rd


def getRegion_RCvec(bamfile, region):
    rlen = region[2] - region[1]
    region_rc = np.zeros(rlen, dtype=DTYPE)
    reads = bamfile.fetch(region[0], region[1], region[2])
    for r in reads:
        if r.pos < r.mpos and r.pos >= region[1] and r.pos < region[2]:
            if r.mapping_quality >= config.DATABASE['mapq_threshold']:
                region_rc[(r.pos - region[1])] += 1

    return region_rc


def printBasicBamInfo(bamfile):
    logger.info('==============================================')
    logger.info(('- Basic information for bam file: \n {0}').format(bamfile.filename))
    logger.info(('- Mapped {0}').format(bamfile.mapped))
    logger.info(('- Unmapped {0}').format(bamfile.unmapped))
    logger.info('==============================================')

def printBasicBamInfo_print(bamfile):
    print('==============================================')
    print('- Basic information for bam file: \n %s' %(bamfile.filename))
    print('- Mapped %d' %(bamfile.mapped))
    print('- Unmapped %d' %(bamfile.unmapped))
    print('==============================================')

if __name__ == '__main__':
    bamfileName = '/Users/yaozhong/my_work/2019_Projects/1_UNet_IntraSV/201911/data/Sim-A_30x_para.bam'
    bamfile = pysam.AlignmentFile(bamfileName, 'rb')
    printBasicBamInfo_print(bamfile)
    
    """
    region = ('1', 144710310, 144710362)
    rgs = [region]
    vec = getRegion_RDVec_list_worker((bamfileName, [region]))
    print np.max(vec)
    """
    

# okay decompiling data_bam2rd.pyc
