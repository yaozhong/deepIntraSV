#########################################################################
# File Name: 1.hyperParameter.sh
# Author: Yao-zhong Zhang
# mail: yaozhong@ims.u-tokyo.ac.jp
# Created Time: Fri Jan  4 21:08:19 2019
#########################################################################
#!/bin/bash

# this function is used to tunning hyper paramters

if [ $# -lt 4 ];
then echo "sh 1.hyperParamter.sh DATA MODEL GPU MODEL_PARAM"
exit -1
fi


DATA=$1
MODEL=$2
GPU=$3
MODEL_PARAM=$4
ANNO=$5

LOG_FOLD="../experiment/expLOG"
OUTPUT="$LOG_FOLD/2.CV_eval_${DATA}_${MODEL}_${ANNO}.txt"

echo "Start the CV evluation on NA12878 data for different read depth ...\n $MODEL_PARAM " > $OUTPUT

echo "python train.py -b 1000 -em single -ds CV -da 0 -d $DATA -m $MODEL -g $GPU  -mp $MODEL_PARAM" >> $OUTPUT


for i in 1 2 3 4 5
do	
	START=$(date +%s)
	python train.py -b 1000 -em single -ds CV -da 0 -d $DATA -m $MODEL -g $GPU  -mp $MODEL_PARAM >> $OUTPUT  2>&1
	END=$(date +%s)
	DIFF=$(($END - $START))

	echo "\n@ TRAIL ${i} takes $DIFF seconds\n" >> $OUTPUT
done
