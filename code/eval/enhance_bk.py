"""
Data:   2020-01-14
Author: Yao-zhong Zhang @ IMSUT
Enhance the break point resolution for given SVs : VCF or BED
"""

import warnings,sys, os, re
if not sys.warnoptions:
    warnings.simplefilter("ignore")
sys.path.insert(0, '../code')

from dataProcess.data_vcf_parser import *
from dataProcess.data_genBinData import *
from dataProcess.data_genomeStats import *

from keras import models
from models.losses import *
from models.model_baseline import binary_eval, get_new_breakpoint2
from vcf_cmp import compare_rgs_list, three_compare_rgs_list, enhancement_analysis

def enhance_sv(vcf_file, vcf_gold, bam_file, model_pm, bk_dataPath, exclude_rgs_file=None):

	print("[1]. Loading VCF file [%s]" %(vcf_file))
	_, _, sv_list = parse_sim_data_vcf(vcf_file, 0, False, True)

	print("[2]. Preparing RD data for bounary regions ...")
	data = genData_from_rg_list(sv_list, bam_file,  dataAug=0, bk_in=False)

	if(exclude_rgs_file != None):
		exlcude_rgs = load_train_rgs(exclude_rgs_file)
		print("Exclude the %d training" %len(exlcude_rgs))
	else:
		exlcude_rgs = None
		print("No exlucde rg is applied!")

	# original dist, not used in teh function
	rgList = data["rgList"]

	# global statistics loading
	if not os.path.exists(bk_dataPath):
		cache_genome_statistics(bam_file, bk_dataPath, config.DATABASE["genomeSampleRate"])	
	m_rd, std_rd, md_rd, gc_mrd_table = load_genome_statistics(bk_dataPath)

	#visal_rd_genome(bk_dataPath, False)

	# nomralization the data and genrate the mapping
	x_data, y_data = np.array(data["x"]), np.array(data["y"])
	x_data = (x_data - m_rd)/std_rd
	x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], 1)

	print("\n[3]. Deep segmentation model segment for the region ...")
	print("* Loading model parameters...")
	model = models.load_model(model_pm, custom_objects={'abc_dice_loss':abc_dice_loss, 'dice_loss': dice_loss, 'dice_coef':dice_coef, \
		'iou_score':iou_score})
	
	logits = model.predict(x_data, verbose=1)
	pred = (logits > 0.5).astype(np.float32).reshape(logits.shape[0], logits.shape[1])
	gold = y_data.astype(np.float32)

	# break_point evluation, to check
	print("\n[4]. Evluating and summary results ...")
	old_rg_list, new_rg_list = get_new_breakpoint2(x_data, data["rgs"], y_data, pred, "null", False)

	# added 2020-03-16 
	print("\n[4.1]. Print the post-processing for the unproper shift that make SV short than [50] ...")
	num_of_invalid = 0
	for i in range(len(new_rg_list)):
		if(new_rg_list[i][2] - new_rg_list[i][1] < 50):
			new_rg_list[i] = old_rg_list[i]
			num_of_invalid += 1
	print("[INFO]: UNPROPOER shift [%d] that makes SV length less than 50, are replaced it back!"  %(num_of_invalid))

	if vcf_gold != None:
		print("\n[5]. Post processing and evluated with gold SV ...")
		# load the gold standard VCF, note not filtering for the gold standard SV
		_, _, sv_gold = parse_sim_data_vcf(vcf_gold, 0, False, False)

  		figureSavePath= config.DATABASE["enhance_output_fold"] + date.today().strftime("%Y%m%d") + "_"
  		print("[##] Segmentaiton and break point:")
  		evluation_breakpoint(x_data, np.array(data["rgs"]), gold, pred, figureSavePath + "_bk_debug", False) 	

  		# check the evluation
  		enhancement_analysis(new_rg_list, old_rg_list, sv_gold, exlcude_rgs, False)

		## all evaluation
		print("\n############### Overall Evluation ################")
		print("[1] Original distance:")
		compare_rgs_list(old_rg_list, sv_gold, exlcude_rgs, False)

		print("\n[2] Enhanced distance:")
		compare_rgs_list(new_rg_list, sv_gold, exlcude_rgs, False)

	print("\n############### Saving the enhancement result ################")
	# analysis results according to SV_len and SV_type
	file_name_elems = vcf_file.split("/")
	file_name = file_name_elems[-4] + "_" + file_name_elems[-3] + "_" + file_name_elems[-2]
	file_name = file_name.replace("run_sim_20200119", "")
	file_name = re.sub(r'2020\d+_', "", file_name)
	file_name = file_name.replace("Rikan_", "")
	file_name = "bin-" + str(config.DATABASE["binSize"])+ "_" + file_name 

	old_result = open(config.DATABASE["enhance_output_fold"]+ "/old_" + file_name + ".bed", "w")
	new_result = open(config.DATABASE["enhance_output_fold"]+ "/enhanced_" + file_name + ".bed", "w")

	for sv in old_rg_list:
		old_result.write("%s\t%d\t%d\t%s\t0\t0\t0\t0\n" %(sv[0], sv[1]+1, sv[2], sv[3]))
	for sv in new_rg_list:
		new_result.write("%s\t%d\t%d\t%s\t0\t0\t0\t0\n" %(sv[0], sv[1]+1, sv[2], sv[3]))

	old_result.close()
	new_result.close()


def enhance_sv_test(vcf_file, vcf_gold, bam_file, model_pm, bk_dataPath, exclude_rgs_file=None):

	
	print("\n[5]. Post processing and evluated with gold SV ...")
	_, _, sv_gold = parse_sim_data_vcf(vcf_gold, 0, False, False)

	# test split case
	svLen_idx_dic = get_split_len_idx(sv_gold)
	for k in svLen_idx_dic.keys():
		print(k,len(svLen_idx_dic[k]))


def cmd_eval():

	parser = argparse.ArgumentParser(description='BK enhancement for SV augmentation')
	parser.add_argument('--dataSelect', '-d', type=str, default="sample1", required=True, help='sample name')
	parser.add_argument('--bam_file', '-bam', type=str, default="", required=True, help='WGS bam file')
	parser.add_argument('--vcf', '-v', type=str, default="", required=True, help='initial vcf file')
	parser.add_argument('--vcf_ci', '-ci', type=int, default=99999999, required=False, help='upbound of breakpoint confidence interval for filtering')
	parser.add_argument('--vcf_gold', '-vg', type=str, default=None, required=False, help='initial vcf file')	

	parser.add_argument('--bin', '-b', type=int, default=400, required=True, help='bin size of target region')
	parser.add_argument('--genomeStat', '-gs', type=str, default="", required=True, help='background RD statistic files')
	parser.add_argument('--model', '-mp', type=str, default="", required=True, help='model hyperparameter file')
	parser.add_argument('--exRgs', '-er', type=str, default=None, required=False, help='exclude regions')

	parser.add_argument('--out_fold', '-o', type=str, default="./", required=True, help='Results output fold')

	parser.add_argument('--gpu', '-g', type=str, default="0", required=False, help='assign the task GPU')

	args = parser.parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

	config.DATABASE["fix_center"] = True
	config.DATABASE["binSize"] = args.bin
	config.DATABASE["vcf_ci"] = args.vcf_ci

	config.DATABASE["enhance_output_fold"] = args.out_fold
	config.DATABASE["heatmap_fold"] = args.out_fold

	vcf_file = args.vcf
	vcf_gold = args.vcf_gold
	train_rgs = args.exRgs

	bam_file = args.bam_file
	bk_dataPath = args.genomeStat
	model_pm = args.model

	enhance_sv(vcf_file, vcf_gold, bam_file, model_pm, bk_dataPath, train_rgs)


if __name__ == "__main__":

	cmd_eval()

