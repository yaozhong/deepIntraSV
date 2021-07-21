"""
Date : 2018-06-22
Author: Yao-zhong Zhang
Description: loading the bigwig mappability files and prepare the data for the input. 
"""

from util import *

"""
get the bin sized vectors for the given regions
"""
def getMappability(region, binSize, fileName):

	with pyBigWig.open(fileName) as mapB:

		nSeg = (region[2] - region[1])/binSize
		map_vec = np.zeros(nSeg, np.float32)

		for i in range(nSeg):
			rg = (region[0], region[1] + i*binSize, region[1] + (i+1)*binSize)
			map_vec[i] = np.mean(mapB.values(rg[0], rg[1], rg[2]))
	return map_vec


def getMappability_worker(params):

	region, binSize, fileName = params

	with pyBigWig.open(fileName) as mapB:

		nSeg = (region[2] - region[1])/binSize
		map_vec = np.zeros(nSeg, np.float32)

		for i in range(nSeg):
			rg = (region[0], region[1] + i*binSize, region[1] + (i+1)*binSize)
			map_vec[i] = np.mean(mapB.values(rg[0], rg[1], rg[2]))
	return map_vec


def getMappability_worker_nobin(params):

	rgs, fileName = params
        map_vec = np.zeros(len(rgs), np.float32)

	with pyBigWig.open(fileName) as mapB:
    
            for i, region in enumerate(rgs):
                region = normChrName(region, 1)
                try:
                    vals = mapB.values(region[0], region[1], region[2])
                except:
                    map_vec[i] = 0
                else:
                    map_vec[i] = np.mean(vals)

        return map_vec


## parallel version of getting mappability

#no_bin case
def getRegions_mappability_parallel_nobin(fileName, rgs):

    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)

    if len(rgs) < cores:
        cores = len(rgs)
    
    # core based split
    step = int(len(rgs)/cores)
    rgsList = [ rgs[i*step: (i+1)*step] for i in range(cores-1)]
    rgsList.append( rgs[(cores-1)*step:])

    todo_params = [(rs, fileName) for rs in rgsList ]
    output = pool.map(getMappability_worker_nobin, todo_params)

    pool.close()
    pool.join()

    vec = np.concatenate(output)

    return vec

# the bin size is used for segmenting the given region
def getChr_mappability_parallel_pool(chr_rg, binSize, fileName):

	cores = multiprocessing.cpu_count()
	pool = multiprocessing.Pool(processes=cores)

	# segment 
	nSegs = (chr_rg[2] - chr_rg[1])/binSize
	nThreadUnit = nSegs/cores

	# core based data split
	todo_params = []
	for c in range(cores):

		start_local = c * nThreadUnit * binSize
		end_local = (c+1)* nThreadUnit * binSize
		rg = (chr_rg[0], start_local, end_local)
		todo_params.append((rg, binSize, fileName))

	end_position = nThreadUnit* cores * binSize
	if  chr_rg[2] > end_position:
		rg = (chr_rg[0], end_position, chr_rg[2])
		todo_params.append((rg,binSize, fileName))

	# pool calcing
	output = pool.map(getMappability_worker, todo_params)

	pool.close()
	pool.join()

	# Join the split segs ... 
	vec = np.concatenate(output)
	return vec


def testing():

	filePath = config.DATABASE["mappability_file"]
	rg = ("chr1", 1, 247199719)
	vec = getChr_mappability_parallel_pool(rg, 1000, filePath)
	print vec.shape

"""
todo issues:  1.chromeName checking and standardization
    		  2.mappingbility issues.           
"""
if __name__ == "__main__":
	testing()


