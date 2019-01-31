#########################################################################
# File Name: 1.hyperParameter.sh
# Author: Yao-zhong Zhang
# mail: yaozhong@ims.u-tokyo.ac.jp
# Created Time: Fri Jan  4 21:08:19 2019
#########################################################################
#!/bin/bash

# this function is used to tunning hyper paramters

if [ $# -lt 4 ];
then echo "sh 5.dataAugmentation.sh DATA MODEL MODEL_PARAM GPU"
exit -1
fi

DATA=$1
MODEL=$2
MODEL_PARAM=$3
GPU=$4
ANNO=$5

LOG_FOLD=../experiment/expLOG/
OUTPUT=$LOG_FOLD/5.dataAug_${DATA}_${MODEL}_${ANNO}.txt
echo "Evluating the peformance of data Augmentation" > $OUTPUT


for i in 1 2 3 4 5
do
	START=$(date +%s)
	python train.py -b 1000 -em single -ds StratifyNew -da 10 -d $DATA -m $MODEL -g $GPU  -mp $MODEL_PARAM >> $OUTPUT 2>&1 
	END=$(date +%s)
	DIFF=$(($END - $START))
	echo "\n@ TRAIL ${i} takes $DIFF seconds" >> $OUTPUT
done

echo "*************** None data augmentation run ************" >> $OUTPUT


for i in 1 2 3 4 5
do
	START=$(date +%s)
	python train.py -b 1000 -em single -ds StratifyNew -da 0 -d $DATA -m $MODEL -g $GPU  -mp $MODEL_PARAM >> $OUTPUT 2>&1 
	END=$(date +%s)
	DIFF=$(($END - $START))
	echo "\n@ TRAIL ${i} takes $DIFF seconds" >> $OUTPUT
done
