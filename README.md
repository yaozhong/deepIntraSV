# deepIntraSV
DeepIntraSV is a U-net based model used for detecting SVs inside of a bin with base-pair read-depth (RD) inforamtion.
More details can be found in https://doi.org/10.1101/503649

## Docker enviorment
We provide a docker image for running this code
```
docker pull yaozhong/deep_intra_sv
```
* ubuntu 14.04.4
* Tensorflow 1.8.0
* Keras 2.2.4

## Configuration file
The parameters are required to be assigned in the config.py file.
Several frequent parameters can be also changed through command line parameters.

### Required files
1. reference files are located in ./data/reference/, which includes required reference files: 
* reference genome fa file and index (Please download from http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/)
* mappability of 100-mer
* Encode hg19 blacklist regions
* hg19 Chromesome length 

2. SV annotation files./data/SV_annotation/
  (a python script is provided to parse VCF for SV regions)

## Data pre-processing and caching
A multi-core version of pysam is applied. In default, all cores will be used 
to generate RD bin files from bam file. For each training data, 
background statistics of RD are first calcuated through sampling the data.


## Training
Input are WGS bam file(s). Cached data will be first searched according to current parameters,
If cache files are not found, the code will process the bam file and cache the data.
The cached files are saved in ./data/data_cache/
Output are the saved model weights.

Users can change manually change the model parameter file.
If no model parameter file is provided, the code will use hyperOpt to search preDefined hyper-parameter spaces.

```
python train.py -b 1000 -em single -ds StratifyNew -d na12878_60x -da 0 -m UNet -g 0 -mp ../experiment/model_param/unet_default
```


## Testing
```
python test.py -b 1000  -em single -ds StratifyNew -d na12878_60x -m UNet -mw ../experiment/model/na12878_60x_RD_bin1000_TRAIN_extendContext-0_dataAug-0_filter-BQ30-MAPQ-30_AnnoFile-annoFile\:NA12878_1kGP_IC--1.bed\|UNet_maxpoolingLen_5-5-2-2-5-5-convWindowLen_7-lr_0.001-batchSize64-epoch100-dropout0.2.h5 -mp ../experiment/model_param/unet_default
```

## Experiment logs
For reproducibility, the scripts of related experiments are listed in the ./experiment/expLOG fold.


(*) This document is still under construction and will be updated very soon.

