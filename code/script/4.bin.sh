#########################################################################
# File Name: 1.hyperParameter.sh
# Author: Yao-zhong Zhang
# mail: yaozhong@ims.u-tokyo.ac.jp
# Created Time: Fri Jan  4 21:08:19 2019
#########################################################################
#!/bin/bash

# this function is used to tunning hyper paramters


if [ $# -lt 5 ];
then echo "sh script.sh DATA BIN MODEL MODEL_PARAM GPU"
fi


DATA=$1
BIN=$2
MODEL=$3
MODEL_PARAM=$4
GPU=$5
ANNO=$6

LOG_FOLD=../experiment/expLOG/
OUTPUT=$LOG_FOLD/4.bin_${DATA}_Bin-${BIN}_${MODEL}_${ANNO}.txt

echo "Evluation the bin size effect..." > $OUTPUT


echo ">> Bin = $BIN \n" >> $OUTPUT

for i in 1 2 3 4 5
do
	START=$(date +%s)
	python train.py -b $BIN -em single -ds CV -da 0 -d $DATA -m $MODEL -g $GPU -mp $MODEL_PARAM >>$OUTPUT  2>&1 
	END=$(date +%s)
	DIFF=$(($END - $START))
	echo "\n@ TRAIL ${i} takes $DIFF seconds" >> $OUTPUT
done









