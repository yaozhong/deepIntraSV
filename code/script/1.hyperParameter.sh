#########################################################################
# File Name: 1.hyperParameter.sh
# Author: Yao-zhong Zhang
# mail: yaozhong@ims.u-tokyo.ac.jp
# Created Time: Fri Jan  4 21:08:19 2019
#########################################################################
#!/bin/bash

# this function is used to tunning hyper paramters

if [ $# -ne 3 ];
then echo "sh 1.hyperParamter.sh DATA MODEL GPU"
fi


DATA=$1
MODEL=$2
GPU=$3
ANNO=$4

LOG_FOLD=../experiment/expLOG/
OUTPUT="$LOG_FOLD/1.hyperParameter_${DATA}_${MODEL}_${ANNO}.txt"

echo "python train.py -b 1000 -em single -ds StratifyNew -da 0 -d $DATA -m $MODEL -g $GPU \n" > $OUTPUT 

START=$(date +%s)
python train.py -b 1000 -em single -ds StratifyNew -da 0 -d $DATA -m $MODEL -g $GPU >> $OUTPUT 2>&1 
END=$(date +%s)
DIFF=$(($END - $START))

echo "\n@takes $DIFF seconds"
echo "\n@takes $DIFF seconds" >> $OUTPUT
