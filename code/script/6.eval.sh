#########################################################################
# File Name: 1.hyperParameter.sh
# Author: Yao-zhong Zhang
# mail: yaozhong@ims.u-tokyo.ac.jp
# Created Time: Fri Jan  4 21:08:19 2019
#########################################################################
#!/bin/bash

# this function is used to tunning hyper paramters

if [ $# -ne 3 ];
then echo "sh 6.eval.sh DATA MODEL GPU"
fi

DATA=$1
MODEL=$2
GPU=$3

LOG_FOLD=../experiment/expLOG/

START=$(date +%s)
python test.py -b 1000 -em single -ds StratifyNew -da 0 -d $DATA -m ${MODEL}_test -g $GPU > $LOG_FOLD/6.eval_${MODEL}_test.txt 2>&1 
END=$(date +%s)
DIFF=$(($END - $START))

echo "\n@takes $DIFF seconds"
echo "\n@takes $DIFF seconds" >> $LOG_FOLD/6.eval_${DATA}_${MODEL}.txt
