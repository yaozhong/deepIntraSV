"""
Data:   2020-01-14
Author: Yao-zhong Zhang @ IMSUT
Evluation break points for the query SV and reference SV
"""

import warnings,sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
sys.path.insert(0, '../code')

from dataProcess.data_vcf_parser import *
import config
from util import get_dist_len_idx

import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# add the confidence interval's part, to do one more checking
def compare_rgs_list(sv_list, tsv_list, excldue_list=None, eval_with_ci=False):

	num_overlap = 0
	num_exact, num_exact_both = 0, 0
	num_exact_rough, num_exact_both_rough = 0, 0
	distance = []

	# calcuate the distance distribution (50, 100, 200, 500, 1000, >1000)
	bk_dist_count = [0, 0, 0, 0, 0, 0, 0, 0, 0]
	bk_dist   = [[],[],[],[],[],[],[],[],[]]
	hit_gold_sv_list = []

	# remove the trained rgs used in the evluation
	if excldue_list != None:
		print(" [-] Query rgs=[%d], Gold rgs=[%d], Exclude list=[%d]" %(len(sv_list), len(tsv_list), len(excldue_list)))
		excldue_list = [ "".join(map(str, esv)) for esv in excldue_list]
		tsv_list = [ tsv for tsv in tsv_list if "".join(map(str,tsv)) not in excldue_list]

    # from SV to gSV
	for sv in tqdm(sv_list):

		# updated reuslts 
		if sv[3] in ["INS", "INV", "BND"]:
			continue

		for tsv in tsv_list:
			if( (sv[0] == tsv[0]) and (sv[3] == tsv[3] or (sv[3] in tsv[3]) or (tsv[3] in sv[3])) ):

				# confirm overlapping
				if (sv[1] >= tsv[1] and  sv[1] <= tsv[2]) or  \
				(tsv[1] >= sv[1] and tsv[1] <= sv[2]):

					if eval_with_ci == False:
						overlap = min(tsv[2], sv[2]) - max(tsv[1], sv[1])
						union = (tsv[2] - tsv[1]) + (sv[2] - sv[1]) - overlap

						## gold_sv_len = (tsv[2] - tsv[1] + 1)
					else:
						# not used any more
				   		if tsv[4] <= 0:
				   			left_c = np.abs(tsv[4])
				   		else:
				   			left_c = 0
				   		if tsv[7] >= 0:
				   			right_c = tsv[7]
				   		else:
				   			right_c = 0

				   		overlap = min(tsv[2]+right_c, sv[2]) - max(tsv[1]-left_c, sv[1])
				   		union = (tsv[2] - tsv[1]) + (sv[2] - sv[1]) - overlap + (left_c + right_c)
				   		# overlap rate with reference
				   		## gold_sv_len = (tsv[2] - tsv[1] + 1) + (left_c + right_c)

				   	# this overlap is Jaccard index
				   	overlap_rate = float(overlap) / union

				   	if overlap_rate > config.DATABASE["JS"]:
				   		if ( np.abs(sv[1] - tsv[1]) <= 1 and np.abs(sv[2] - tsv[2]) <= 1):
				   			num_exact_both_rough += 1
				   		elif (np.abs(sv[1] - tsv[1]) <= 1 or np.abs(sv[2] - tsv[2]) <= 1):
				   			num_exact_rough += 1

				   		# exactly match 
				   		if (sv[1] == tsv[1] and sv[2] == tsv[2]):
				   			num_exact_both += 1
				   		elif (sv[1] == tsv[1] or sv[2] == tsv[2]):
				   			num_exact += 1

				   	##################################################

				   	if overlap_rate > config.DATABASE["JS"]:
				   		num_overlap += 1
				   		distance.append(abs(sv[1]-tsv[1]))
				   		distance.append(abs(sv[2]-tsv[2]))

				   		#######################################
				   		# calucate the distance distribution
				   		#######################################
				   		idx = get_dist_len_idx(abs(sv[1]-tsv[1]))
				   		bk_dist_count[idx] += 1
				   		bk_dist[idx].append(abs(sv[1]-tsv[1]))

				   		idx = get_dist_len_idx(abs(sv[2]-tsv[2]))
				   		bk_dist_count[idx] += 1
				   		bk_dist[idx].append(abs(sv[2]-tsv[2]))

				   		# note in this case the number might be larger
				   		hit_gold_sv_list.append(sv)
				   		break

	h_sv_range, h_sv_count = get_split_len_idx(hit_gold_sv_list)
	print("[*] SVs hitted gold has the following distribution:")
	print(",".join(h_sv_range))
	print("	".join([str(s) for s in h_sv_count]))
	
	print("[**] Overlap ones [%d/%d], exatctly ones [%d], exact both ones [%d]" %(num_overlap, len(sv_list), num_exact, num_exact_both))
	print("[**] Relax-1 Overlap ones [%d/%d], exatctly ones [%d], exact both ones [%d]" %(num_overlap, len(sv_list), num_exact_rough, num_exact_both_rough))
	print("[**] Distance is [m=%f, sd=%f]" %(np.mean(distance), np.std(distance)))

	########################################################################################
	print("[**] Distance distributed accroding to length [5,10, 20, 50, 100, 200, 500, 1000, >1000]\n[%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d]"\
		%(bk_dist_count[0], bk_dist_count[1], bk_dist_count[2], bk_dist_count[3], bk_dist_count[4], bk_dist_count[5], bk_dist_count[6], bk_dist_count[7], bk_dist_count[8]))
	print("[**] Distance-mean distributed accroding to length [5,10, 20, 50, 100, 200, 500, 1000, >1000]\n["),
	for bkd in bk_dist:
		if(len(bkd) > 0):
			print(np.round(np.mean(bkd),2)),
		else:
			print("None"),
		print("\t"),
	print("]")

	print("[**] Distance-sd distributed accroding to length [5, 10, 20, 50, 100, 200, 500, 1000, >1000]\n["),
	for bkd in bk_dist:
		if(len(bkd) > 0):
			print(np.round(np.std(bkd), 2)),
		else:
			print("None"),
		print("\t"),
	print("]")


# 2020-03-01
def three_compare_rgs_list(sv_list, sv_list2, tsv_list, excldue_list=None, eval_with_ci=False):

	num_overlap = 0
	num_exact, num_exact_both = 0, 0
	distance = []

	# remove the trained rgs used in the evluation
	if excldue_list != None:
		print(" [-] Query rgs=[%d], Gold rgs=[%d], Exclude list=[%d]" %(len(sv_list), len(tsv_list), len(excldue_list)))
		excldue_list = [ "".join(map(str, esv)) for esv in excldue_list]
		tsv_list = [ tsv for tsv in tsv_list if "".join(map(str,tsv)) not in excldue_list]
		# print("[-] After exlcuding %d" %(len(tsv_list)))

	for i in range(len(sv_list)):

		# updated reuslts 
		if sv_list[i][3] in ["INS", "INV", "BND"]:
			continue

		# ignore the non-changed positions
		if sv_list[i][1] == sv_list2[i][1] and sv_list[i][2] == sv_list2[i][2]:
			continue

		left_dist = np.abs(sv_list[i][1] - sv_list2[i][1])
		right_dist = np.abs(sv_list[i][2] - sv_list2[i][2])

		#query the gold standard ones
		for tsv in tsv_list:

			# start to confirm each condition. stage-1 chr and SV type, in this case, it is all the same for the enhances
			if( (sv_list[i][0] == tsv[0]) and (sv_list[i][3] == tsv[3] or (sv_list[i][3] in tsv[3]) or (tsv[3] in sv_list[i][3])) ):

				overlap_rate1, overlap_rate2 = 0, 0 

				# potential overlapped
				if (sv_list[i][1] >= tsv[1] and  sv_list[i][1] <= tsv[2]) or  (tsv[1] >= sv_list[i][1] and tsv[1] <= sv_list[i][2]):

					# calcuate the overlap
					overlap = min(tsv[2], sv_list[i][2]) - max(tsv[1], sv_list[i][1]) 
					union = (tsv[2] - tsv[1] ) + (sv_list[i][2] - sv_list[i][1] ) - overlap
					overlap_rate1 = float(overlap) / union
				
				if (sv_list2[i][1] >= tsv[1] and  sv_list2[i][1] <= tsv[2]) or  (tsv[1] >= sv_list2[i][1] and tsv[1] <= sv_list2[i][2]):
					# calcuate the overlap
					overlap = min(tsv[2], sv_list2[i][2]) - max(tsv[1], sv_list2[i][1])
					union = (tsv[2] - tsv[1]) + (sv_list2[i][2] - sv_list2[i][1]) - overlap
					overlap_rate2 = float(overlap) / union
			
				
				if(overlap_rate1 > config.DATABASE["JS"] and overlap_rate2 < config.DATABASE["JS"]):
					print("[*] Effective enhancement, overalp:")
					print(sv_list[i])
					print(sv_list2[i])
					print(tsv)
					print("Left_distance=%d, right_distance=%d" %(left_dist, right_dist))
					print("overlap1=%f, overlap2=%f" %(overlap_rate1, overlap_rate2))

					#raw_input("Press any key to continue ...")	

				if(overlap_rate1 < config.DATABASE["JS"] and overlap_rate2 > config.DATABASE["JS"]):
					print("[*] Worse enhancement, overlap:")
					print(sv_list[i])
					print(sv_list2[i])
					print(tsv)
					print("Left_distance=%d, right_distance=%d" %(left_dist, right_dist))
					print("overlap1=%f, overlap2=%f" %(overlap_rate1, overlap_rate2))


				if (sv_list2[i][1] == tsv[1] and sv_list2[i][2] == tsv[2] and (sv_list[i][1] != tsv[1] or sv_list[i][2] != tsv[2])):
					print("[*] worse enhancement, both exactly:")
					print(sv_list[i])
					print(sv_list2[i])
					print(tsv)
					print("Left_distance=%d, right_distance=%d" %(left_dist, right_dist))
					print("overlap1=%f, overlap2=%f" %(overlap_rate1, overlap_rate2))
					#x= raw_input("Press any key to continue ...")	


# 2020-03-10
def enhancement_analysis(new_svs, old_svs, tsv_list, excldue_list=None, eval_with_ci=False):

	num_overlap = 0
	num_exact, num_exact_both = 0, 0
	distance = []

	old_bk_in_range, old_bk_out_range = 0, 0

	# remove the trained rgs used in the evluation
	if excldue_list != None:
		print(" [-] Query rgs=[%d], Gold rgs=[%d], Exclude list=[%d]" %(len(new_svs), len(tsv_list), len(excldue_list)))
		excldue_list = [ "".join(map(str, esv)) for esv in excldue_list]
		tsv_list = [ tsv for tsv in tsv_list if "".join(map(str,tsv)) not in excldue_list]

	shift_matrix = np.zeros((9,9), dtype = int)

	for i in tqdm(range(len(new_svs))):

		# here the type is the same, no changed 
		if new_svs[i][3] in ["INS", "INV", "BND"]:
			continue
		# ignore the non-changed positions
		## 20200406
		if new_svs[i][1] == old_svs[i][1] and new_svs[i][2] == old_svs[i][2]:
			continue

		#query the gold standard ones
		for tsv in tsv_list:

			# start to confirm each condition. stage-1 chr and SV type, in this case, it is all the same for the enhances
			if( (new_svs[i][0] == tsv[0]) and (new_svs[i][3] == tsv[3] or (new_svs[i][3] in tsv[3]) or (tsv[3] in new_svs[i][3])) ):

				overlap_rate1, overlap_rate2 = 0, 0 
				old_gDist_left, new_gDist_left, old_gDist_right, new_gDist_right  = 0,0,0,0

				# calcuate overlap rate
				if (new_svs[i][1] >= tsv[1] and  new_svs[i][1] <= tsv[2]) or  (tsv[1] >= new_svs[i][1] and tsv[1] <= new_svs[i][2]):

					# change this part with considering confidence interval
					if eval_with_ci == False:
						overlap = min(tsv[2], new_svs[i][2]) - max(tsv[1], new_svs[i][1])
						union = (tsv[2] - tsv[1]) + (new_svs[i][2] - new_svs[i][1]) - overlap
						## gold_sv_len = (tsv[2] - tsv[1] + 1)
					else:
				   		if tsv[4] <= 0:
				   			left_c = np.abs(tsv[4])
				   		else:
				   			left_c = 0
				   		if tsv[7] >= 0:
				   			right_c = tsv[7]
				   		else:
				   			right_c = 0

				   		overlap = min(tsv[2]+right_c, new_svs[i][2]) - max(tsv[1]-left_c, new_svs[i][1])
				   		union = (tsv[2] - tsv[1]) + (new_svs[i][2] - new_svs[i][1]) - overlap + (left_c + right_c)

					overlap_rate1 = float(overlap) / union
					#overlap_rate1 = float(overlap) / (new_svs[i][2] - new_svs[i][1] + 1)
					if(new_svs[i][2] - new_svs[i][1] < 50):
						print("\n[DEBUG]: ++++++++++++++++ New shifted size less than 50 ++++++++++++++")
						print(new_svs[i])
						print(old_svs[i])
						print(tsv[0], tsv[1], tsv[2], tsv[3])
	
				new_gDist_left = np.abs(new_svs[i][1] - tsv[1])
				new_gDist_right = np.abs(new_svs[i][2] - tsv[2])

				## old SV detection
				if (old_svs[i][1] >= tsv[1] and  old_svs[i][1] <= tsv[2]) or  (tsv[1] >= old_svs[i][1] and tsv[1] <= old_svs[i][2]):
					# change this part with considering confidence interval
					if eval_with_ci == False:
						overlap = min(tsv[2], old_svs[i][2]) - max(tsv[1], old_svs[i][1])
						union = (tsv[2] - tsv[1]) + (old_svs[i][2] - old_svs[i][1]) - overlap
					else:
				   		if tsv[4] <= 0:
				   			left_c = np.abs(tsv[4])
				   		else:
				   			left_c = 0
				   		if tsv[7] >= 0:
				   			right_c = tsv[7]
				   		else:
				   			right_c = 0

				   		overlap = min(tsv[2]+right_c, old_svs[i][2]) - max(tsv[1]-left_c, old_svs[i][1])
				   		union = (tsv[2] - tsv[1]) + (old_svs[i][2] - old_svs[i][1]) - overlap + (left_c + right_c)

					overlap_rate2 = float(overlap) / union
					#overlap_rate2 = float(overlap) / (old_svs[i][2] - old_svs[i][1] + 1)

				old_gDist_left = np.abs(old_svs[i][1] - tsv[1])
				old_gDist_right = np.abs(old_svs[i][2] - tsv[2])

				# 2020-03-18 add additional count 
				if(overlap_rate2 > config.DATABASE["JS"]):
					# left 
					if old_gDist_left <= 50:
						old_bk_in_range += 1
					else:
					 	old_bk_out_range += 1

					# right
					if old_gDist_right <= 50:
						old_bk_in_range += 1
					else:
					 	old_bk_out_range += 1

				# checking the bounary shift
				if(overlap_rate1 > config.DATABASE["JS"] and overlap_rate2 <= config.DATABASE["JS"]):
					print("[*] -*- Effective -*- enhancement, add overalp regions [new, old, gold]:")
					print(new_svs[i])
					print(old_svs[i])
					print(tsv[0], tsv[1], tsv[2], tsv[3])
					print("overlap1=%f, overlap2=%f" %(overlap_rate1, overlap_rate2))


				if(overlap_rate1 <= config.DATABASE["JS"] and overlap_rate2 > config.DATABASE["JS"]):
					print("[*] -X- Worse enhancement -X-, lose overlap [new, old, gold]:")
					print(new_svs[i])
					print(old_svs[i])
					print(tsv[0], tsv[1], tsv[2], tsv[3])
					print("overlap1=%f, overlap2=%f" %(overlap_rate1, overlap_rate2))

				# this conditional needs checked
				## 20200406
				if(overlap_rate1 <= config.DATABASE["JS"] and overlap_rate2 <= config.DATABASE["JS"]):
					continue

				# checking the count matrix
				old_idx = get_dist_len_idx(old_gDist_left)
				new_idx = get_dist_len_idx(new_gDist_left)
				shift_matrix[old_idx, new_idx] += 1

				old_idx = get_dist_len_idx(old_gDist_right)
				new_idx = get_dist_len_idx(new_gDist_right)
				shift_matrix[old_idx, new_idx] += 1

				break

	print("[INFO+]: Original prediction gold in size of the bin [%d], outside of the bin [%d]." %(old_bk_in_range, old_bk_out_range))
	print(shift_matrix)
	# save the figure to the plot
	print("Plot the matrix ...")
	fig = plt.figure(figsize=(10, 10))	
	ax = sns.heatmap(shift_matrix, annot=True, fmt="d", cmap='OrRd')
	ax.set_xticklabels(['<5', '[5,10)', '[10, 20)', '[20, 50)', '[50,100)', '[100,200)', '[200,500)', '[500,1000)', ">=1000"], rotation=45) 
	ax.set_yticklabels(['<5', '[5,10)', '[10, 20)', '[20, 50)', '[50,100)', '[100,200)', '[200,500)', '[500,1000)', ">=1000"], rotation=45) 

	ax.set_xlabel('After enahancement')  
	ax.set_ylabel('Before enahancement')  
	ax.xaxis.set_ticks_position('top')
	ax.xaxis.set_label_position('top')

	savePath = config.DATABASE["heatmap_fold"] + "/" + datetime.now().strftime("%Y%m%d-%H_%M_%S") + "_enhancment_heatmap.png"
	print("* Heatmap saved in: %s" %(savePath))
	plt.savefig(savePath)

		
def compareSV(vcf1, vcf2):

	_, _, sv_vcf1 = parse_sim_data_vcf(vcf1, 0, False, False)
	_, _, sv_vcf2 = parse_sim_data_vcf(vcf2, 0, False, False)

	num_overlap = 0
	distance = []

	for sv in tqdm(sv_vcf1):

		if sv["sv_type"] not in ["DEL", "DUP", "CNV"]:
			continue

		for tsv in sv_vcf2:
			if(sv["chr"] == tsv["chr"] and sv["sv_type"] == tsv["sv_type"]):
				# confirm overlapping
				if (sv["start"] >= tsv["start"] and  sv["start"] <= tsv["end"]) or  \
				    (tsv["start"] >= sv["start"] and tsv["start"] <= sv["end"]):

					overlap = min(tsv["end"], sv["end"]) - max(tsv["start"], sv["start"])
					union = (tsv["end"] - tsv["start"] ) + (sv["end"] - sv["start"]) - overlap

					overlap_rate = float(overlap) / union
					if overlap_rate > config.DATABASE["JS"]:
						num_overlap += 1
						distance.append(abs(sv["start"]-tsv["start"]))
						distance.append(abs(sv["end"]-tsv["end"]))

	print("[**] Overlap [%d/%d]" %(num_overlap, len(sv_vcf1)))
	print("[**] Distance is [m=%f, sd=%f]" %(np.mean(distance), np.std(distance)))

############################################################
# 20200306, check how many gold is hited in the predicted set.
############################################################
def gold_hit_check(gsv_list, sv_list, eval_with_ci=False):

	sv_gold_hit = []
	bp_exact_hit, bp_partial_hit = [],[]

	for gsv in tqdm(gsv_list):
		if gsv[3] in ["INS", "INV", "BND"]:
			continue

		for sv in sv_list:
			
			# chr and SV type the same
			if( (sv[0] == gsv[0]) and (sv[3] == gsv[3] or (sv[3] in gsv[3]) or (gsv[3] in sv[3])) ):
				overlap_rate = 0

				# in the range
				if (sv[1] >= gsv[1] and  sv[1] <= gsv[2]) or  (gsv[1] >= sv[1] and gsv[1] <= sv[2]):
					
					if eval_with_ci == False:
						overlap = min(gsv[2], sv[2]) - max(gsv[1], sv[1])
						union = (gsv[2] - gsv[1] ) + (sv[2] - sv[1] ) - overlap
					else:
				   		if gsv[4] <= 0:
				   			left_c = np.abs(gsv[4])
				   		else:
				   			left_c = 0
				   		if gsv[7] >= 0:
				   			right_c = gsv[7]
				   		else:
				   			right_c = 0

				   		overlap = min(gsv[2]+right_c, sv[2]) - max(gsv[1]-left_c, sv[1])
				   		union = (gsv[2] - gsv[1] ) + (sv[2] - sv[1]) - overlap + (left_c + right_c)
					
					overlap_rate = float(overlap) / union

					if(overlap_rate > config.DATABASE["JS"]):

						#check the boundary exactness
						if ( (np.abs(sv[1] - gsv[1])) <= 1 and (np.abs(sv[2] - gsv[2]) <= 1) ):
							bp_exact_hit.append(sv)
				 	  	elif ((np.abs(sv[1] - gsv[1]) <= 1) or (np.abs(sv[2] - gsv[2]) <= 1)):
				 	  		bp_partial_hit.append(sv)

						sv_gold_hit.append(sv)

						break

	return sv_gold_hit, bp_exact_hit, bp_partial_hit
					
# comparing the overlap
def overlap_checking(sv_list_file, sv_list2_file, tsv_list_file, exclude_rgs_file=None, eval_with_ci=False):

	_, _, sv_list  = parse_sim_data_vcf(sv_list_file, 0, False, True)
	_, _, sv_list2 = parse_sim_data_vcf(sv_list2_file, 0, False, True)
	_, _, tsv_list = parse_sim_data_vcf(tsv_list_file, 0, False, False)

	if(exclude_rgs_file != None):
		excldue_list = load_train_rgs(exclude_rgs_file)
	else:
		excldue_list = None
	if excldue_list != None:
		excldue_list = [ "".join(map(str, esv)) for esv in excldue_list]
		tsv_list = [ tsv for tsv in tsv_list if "".join(map(str,tsv)) not in excldue_list]
		print("[-] Gold SVs in the evluation is %d" %(len(tsv_list)))

	# note this part is relatively slow
	sv1_gold_hit,_ ,_ = gold_hit_check( tsv_list, sv_list,   eval_with_ci)
	sv2_gold_hit,_ ,_ = gold_hit_check( tsv_list, sv_list2,  eval_with_ci)

	# checking the overlapped ogld
	num_gold_overlap = 0
	union_gold_sv = []

	for sv in sv1_gold_hit:
		for tsv in sv2_gold_hit:
			if sv[0]==tsv[0] and sv[1]==tsv[1] and sv[2]==tsv[2] and sv[3]==tsv[3] :
				union_gold_sv.append(sv)
				break

	print("\n[!]: SV1_hit_gold=%d, SV2_hit_gold=%d and overlap of predictions=%d" %(len(sv1_gold_hit), len(sv2_gold_hit), len(union_gold_sv)))
	
	# 2. checking the SV length distribution
	sv_range, sv_count = get_split_len_idx(union_gold_sv)
	print("[!] The SV distribution in the overlapped gold is:")
	print(",".join(sv_range))
	print("	".join([str(s) for s in sv_count]))

# SVs with exactly boundary checkking

# comparing the overlap
def exact_checking(sv_list_file, sv_list2_file, tsv_list_file, exclude_rgs_file=None, eval_with_ci=False):

	_, _, sv_list  = parse_sim_data_vcf(sv_list_file, 0, False, True)
	_, _, sv_list2 = parse_sim_data_vcf(sv_list2_file, 0, False, True)
	_, _, tsv_list = parse_sim_data_vcf(tsv_list_file, 0, False, False)

	if(exclude_rgs_file != None):
		excldue_list = load_train_rgs(exclude_rgs_file)
	else:
		excldue_list = None

	if excldue_list != None:
		excldue_list = [ "".join(map(str, esv)) for esv in excldue_list]
		tsv_list = [ tsv for tsv in tsv_list if "".join(map(str,tsv)) not in excldue_list]
		print("[-] Gold SVs in the evluation is %d" %(len(tsv_list)))

	# note this part is relatively slow, the first one is gold
	## sv1_gold_hit, sv1_exact_hit, sv1_partial_hit = gold_hit_check( tsv_list, sv_list,   eval_with_ci)
	## sv2_gold_hit, sv2_exact_hit, sv2_partial_hit = gold_hit_check( tsv_list, sv_list2,  eval_with_ci)
	
	## 20200407
	sv1_gold_hit, sv1_exact_hit, sv1_partial_hit = gold_hit_check( sv_list, tsv_list, eval_with_ci)
	sv2_gold_hit, sv2_exact_hit, sv2_partial_hit = gold_hit_check( sv_list2,tsv_list, eval_with_ci)

	# checking the overlapped ogld
	num_gold_overlap = 0
	union_gold_sv, union_exact_sv, union_partial_sv = [],[],[]
	for sv in sv1_gold_hit:
		for tsv in sv2_gold_hit:
			if sv[0]==tsv[0] and sv[1]==tsv[1] and sv[2]==tsv[2] and sv[3]==tsv[3] :
				union_gold_sv.append(sv)
				break		
	print("\n[Total]: SV1_hit_gold=%d, SV2_hit_gold=%d and overlap of predictions=%d" %(len(sv1_gold_hit), len(sv2_gold_hit), len(union_gold_sv)))
	
    ##########################################################
	for sv in sv1_exact_hit:
		for tsv in sv2_exact_hit:
			if sv[0]==tsv[0] and sv[1]==tsv[1] and sv[2]==tsv[2] and sv[3]==tsv[3] :
				union_exact_sv.append(sv)
				break
	print("\n[Exact boundary]: SV1_hit_gold=%d, SV2_hit_gold=%d and overlap of predictions=%d" %(len(sv1_exact_hit), len(sv2_exact_hit), len(union_exact_sv)))

		# checking
	checking_sv = []
	for sv in sv1_exact_hit:
		for tsv in sv2_gold_hit:
			if sv[0]==tsv[0] and sv[1]==tsv[1] and sv[2]==tsv[2] and sv[3]==tsv[3]:
				checking_sv.append(sv)
				break		
	print("[Exact-all query]: SV1_hit_gold=%d, SV2_hit_gold=%d and overlap of predictions=%d" %(len(sv1_exact_hit), len(sv2_gold_hit), len(checking_sv)))
	
	###########################################################

	for sv in sv1_partial_hit:
		for tsv in sv2_partial_hit:
			if sv[0]==tsv[0] and sv[1]==tsv[1] and sv[2]==tsv[2] and sv[3]==tsv[3] :
				union_partial_sv.append(sv)
				break
	print("\n[Partial boundary]: SV1_hit_gold=%d, SV2_hit_gold=%d and overlap of predictions=%d" %(len(sv1_partial_hit), len(sv2_partial_hit), len(union_partial_sv)))

	checking_sv = []
	for sv in sv1_partial_hit:
		for tsv in sv2_gold_hit:
			if sv[0]==tsv[0] and sv[1]==tsv[1] and sv[2]==tsv[2] and sv[3]==tsv[3]:			
				checking_sv.append(sv)
				break	
	print("[Partial-all query]: SV1_hit_gold=%d, SV2_hit_gold=%d and overlap of predictions=%d" %(len(sv1_partial_hit), len(sv2_gold_hit), len(checking_sv)))

	#############################################################

	
	# 2. checking the SV length distribution
	sv_range, sv_count = get_split_len_idx(union_gold_sv)
	print("\n[!] The SV distribution in the overlapped gold is:")
	print(",".join(sv_range))
	print("	".join([str(s) for s in sv_count]))

	sv_range, sv_count = get_split_len_idx(union_exact_sv)
	print("[Exact BP] The SV distribution in the overlapped gold is:")
	print(",".join(sv_range))
	print("	".join([str(s) for s in sv_count]))

	sv_range, sv_count = get_split_len_idx(union_partial_sv)
	print("[Partial BP] The SV distribution in the overlapped gold is:")
	print(",".join(sv_range))
	print("	".join([str(s) for s in sv_count]))


if __name__ == "__main__":

	parser = argparse.ArgumentParser("Compare two SVs with VCF or bed files...")

	parser.add_argument('--vcf1', '-vcf1', type=str, default="", required=True, help="vcf/bed annotation")
	parser.add_argument('--vcf2', '-vcf2', type=str, default="", required=True, help="reference vcf file")
	parser.add_argument('--gvcf', '-gvcf', type=str, default="", required=True, help="reference vcf file")
	parser.add_argument('--exFile', '-ex', type=str, default=None, required=True, help="reference vcf file")

	args = parser.parse_args()
	#overlap_checking(args.vcf1, args.vcf2, args.gvcf, args.exFile , True)
	exact_checking(args.vcf1, args.vcf2, args.gvcf, args.exFile , False)




