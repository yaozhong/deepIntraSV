#########################################################################
# File Name: 1.hyperParameter.sh
# Author: Yao-zhong Zhang
# mail: yaozhong@ims.u-tokyo.ac.jp
# Created Time: Fri Jan  4 21:08:19 2019
#########################################################################
#!/bin/bash

# this function is used to tunning hyper paramters

if [ $# -lt 4 ];
then echo "sh 6.crossSample DATA DATA2 MODEL GPU MODEL_PARAM"
exit -1
fi


DATA=$1
DATA2=$2
MODEL=$3
MODEL_PARAM=$4
GPU=$5
ANNO=$6

LOG_FOLD="../experiment/expLOG"
OUTPUT="$LOG_FOLD/7.crossSample_${DATA}_${DATA2}_${MODEL}_${ANNO}.txt"

echo "*Cross sample evlauation...\n $MODEL_PARAM " > $OUTPUT
echo "*python train.py -b 1000 -em cross -ds cross -da 0 -d $DATA -d2 $DATA2 -m $MODEL -g $GPU -mp $MODEL_PARAM" >> $OUTPUT

# take 5 time run for the model na12878_train test on the other
for i in 1 2 3 4 5
do	
	START=$(date +%s)
	python train.py -b 1000 -em cross -ds cross -da 0 -d $DATA -d2 $DATA2 -m $MODEL -g $GPU -mp $MODEL_PARAM >> $OUTPUT  2>&1
	END=$(date +%s)
	DIFF=$(($END - $START))

	echo "\n@ TRAIL ${i} takes $DIFF seconds\n" >> $OUTPUT
done
