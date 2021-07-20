"""
Date: 2018-11-05
Description: loading and standardized CNV annotation data.

2018-12-15 add the confidence information to the region.

chr start   end type    start_ci_l  start_ci_r  end_ci_l    end_ci_r    rgLen
"""

import re
from util import normChrStrName

"""
processing files with TAB seperation and first line (start with #) of field names,
chr, start, end is the mustbe containning field.
return regionList
"""
def load_region_file(file_name, cnvType=None):

	# default setttings
	names=["chr", "start", "end"]
	num_elems=3
	typeName= cnvType
	idx_dic = {}
	rgList = []

	with open(file_name, 'r') as inFile:

		for line in inFile:
			if line.startswith("#"):
				line = line.replace("#", "")
				names = line.strip().split("\t") 
				
				for i, name in enumerate(names):
					idx_dic[name.lower()] = i

				num_elems = len(names)
				continue

			line = line.strip()
			elems = line.split("\t")
			assert(len(elems) == num_elems)

			if len(elems) > 3:
				typeName = elems[idx_dic["type"]]
			chrName = normChrStrName(elems[idx_dic["chr"]], 0)
			start = int(elems[idx_dic["start"]])
			end = int(elems[idx_dic["end"]])

                        if len(elems) > 7:
                            start_ci_l = int(elems[idx_dic["start_ci_l"]])
                            start_ci_r = int(elems[idx_dic["start_ci_r"]])
                            end_ci_l = int(elems[idx_dic["end_ci_l"]])
                            end_ci_r = int(elems[idx_dic["end_ci_r"]])
                        else:
                            start_ci_l, start_ci_r, end_ci_l, end_ci_r = 1, -1, 1, -1

			rg = (chrName, start, end, typeName, start_ci_l, start_ci_r, end_ci_l, end_ci_r, end-start)
			rgList.append(rg)

	return rgList

##################################################################
def genBkDic(rgs):

    bkDic = {}
    for rg in rgs:
        if rg[0] not in bkDic.keys():
            bkDic[rg[0]] = []

        bkDic[rg[0]].append(int(rg[1]))
        bkDic[rg[0]].append(int(rg[2]))

    for ch in bkDic.keys():
        bkDic[ch] = sorted(bkDic[ch])

    return bkDic

#################################################################

if __name__ == "__main__":

	file_name = "../data/CNV_annotation/mills_nature/NA12878_ref_hg36"

	onekg_rgs = load_region_file(file_name)

        for rg in onekg_rgs:
            print("%s\t%d\t%d\t%s" %(normChrStrName(rg[0],1), rg[1], rg[2], rg[3]))







