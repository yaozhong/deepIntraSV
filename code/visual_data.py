# testing different features extracted by pysam

import pysam

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os, shutil

from vcf_parser import parse_sim_data_vcf

# Visulaiztion
# visual input data, and get the intution for develpment your algorithm 
def visual_data(bam_file, vcf_file, output_dir="../experiment/visual_na12878"):

	ext = 500
	train_sv, test_sv = parse_sim_data_vcf(vcf_file, 0.1, False)

	sv_type_group = test_sv.groupby("sv_type").count()
	sv_types = sv_type_group.index.tolist()
	
	# output direction
	output_dir += "_ext-" + str(ext) + "/"
	if(os.path.isdir(output_dir)):
		shutil.rmtree(output_dir, ignore_errors=False, onerror=None)
	os.mkdir(output_dir)
	for svt in sv_types:
		os.mkdir(output_dir + svt)

	samfile = pysam.AlignmentFile(bam_file, "rb")

	for index, sv in test_sv.iterrows():
		svt = sv["sv_type"]
		rgs = (sv["chr"], sv["start"], sv["end"])
		coverage = get_coverage(samfile, rgs, ext)

		## new input feature testing part
		fig, ax1 = plt.subplots(figsize=(10,5))
		plt.scatter(x=range(len(coverage)), y= coverage)
		plt.axvline(ext, color= "red")
		plt.axvline(rgs[2]-rgs[1] + ext, color="red")
		plt.savefig(output_dir + svt + "/" + "_".join([str(x) for x in rgs]) + ".png")
		plt.close("all")



def get_coverage(samfile, rgs, ext=100):
	
	coverage = np.zeros(rgs[2] -rgs[1] + 2*ext)
	pileup = samfile.pileup(rgs[0], rgs[1]- ext, rgs[2]+ext)
	for pColumn in pileup:
		if pColumn.pos >= rgs[1]-ext and pColumn.pos < rgs[2]+ext:
			#pColumn.set_min_base_quality(config.DATABASE['base_quality_threshold'])
			coverage[pColumn.pos - rgs[1] + ext] = pColumn.get_num_aligned()

	return coverage
	

def target_region_investigation(bam_file):

	rgs=("1", 119482197, 119483618)
	samfile = pysam.AlignmentFile(bam_file, "rb")

	reads = samfile.fetch(rgs[0], rgs[1], rgs[2])
	for read in reads:
	
		print(read.cigarstring)
		print(read.cigartuples)


if __name__ == "__main__":

	# copy the data and visulaization

	bam_file = "../data/simData/Sim-A_30x_para.bam"
	vcf_file = "../data/simData/Sim-A.SV.vcf"


	visual_data(bam_file, vcf_file)
	#target_region_investigation(bam_file)


