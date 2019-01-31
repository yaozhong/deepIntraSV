#########################################################################
# File Name: 1.hyperParameter.sh
# Author: Yao-zhong Zhang
# mail: yaozhong@ims.u-tokyo.ac.jp
# Created Time: Fri Jan  4 21:08:19 2019
#########################################################################
#!/bin/bash

# this function is used to tunning hyper paramters

if [ $# -lt 4 ];
then echo "sh 1.hyperParamter.sh SPLIT MODEL GPU MODELPARAM ANNO"
exit -1
fi

SPLIT=$1
MODEL=$2
GPU=$3
MODEL_PARAM=$4
ANNO=$5

LOG_FOLD="../experiment/expLOG/"
OUTPUT="$LOG_FOLD/3.depth_60x_downSample_${SPLIT}_${MODEL}_${ANNO}"



echo "** Evluation of the 60x downSampling results using CV **" > $OUTPUT

START=$(date +%s)

echo "****DS0.1*****\n" >>$OUTPUT
python train.py -b 1000 -em single -ds $SPLIT -da 0 -d na12878_60x_ds0.1 -m $MODEL -g $GPU  -mp $MODEL_PARAM >> $OUTPUT 2>>$OUTPUT


echo "****DS0.2*****\n" >>$OUTPUT
python train.py -b 1000 -em single -ds $SPLIT -da 0 -d na12878_60x_ds0.2 -m $MODEL -g $GPU  -mp $MODEL_PARAM >> $OUTPUT 2>>$OUTPUT


echo "****DS0.3*****\n" >>$OUTPUT
python train.py -b 1000 -em single -ds $SPLIT -da 0 -d na12878_60x_ds0.3 -m $MODEL -g $GPU  -mp $MODEL_PARAM >> $OUTPUT 2>>$OUTPUT 


echo "****DS0.4*****\n" >>$OUTPUT
python train.py -b 1000 -em single -ds $SPLIT -da 0 -d na12878_60x_ds0.4 -m $MODEL -g $GPU  -mp $MODEL_PARAM >> $OUTPUT 2>>$OUTPUT


echo "****DS0.5*****\n" >>$OUTPUT
python train.py -b 1000 -em single -ds $SPLIT -da 0 -d na12878_60x_ds0.5 -m $MODEL -g $GPU  -mp $MODEL_PARAM >> $OUTPUT 2>>$OUTPUT


echo "****DS0.7*****\n" >>$OUTPUT
python train.py -b 1000 -em single -ds $SPLIT -da 0 -d na12878_60x_ds0.7 -m $MODEL -g $GPU  -mp $MODEL_PARAM >> $OUTPUT 2>>$OUTPUT

END=$(date +%s)
DIFF=$(($END - $START))

echo "\n@Total takes $DIFF seconds" >> $OUTPUT
