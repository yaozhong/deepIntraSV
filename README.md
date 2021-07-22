# Breakpoint enhancement for read-depth based SV callers

RDBKE is a breakpoint enhancment pipeline for read-depth (RD) based SV callers using deep segmenation model UNet.
UNet is used to learn specific patterns of base-wise RDs surrounding known breaking points.
It is designed for in-sample and cross-sample applications.
More details can be found in the manuscript https://doi.org/10.1101/503649

Old branch DeepIntraSV only contains model-level training and testing.
RDBKE branch added the enhancement module for RD-based SV callers (e.g., CNVnator).

* RDBKE overall pipeline
![](figures/Fig1_workflow.png)

* Model structure of UNet used for RDBKE:
![](figures/Fig2_Unet_structure.png)

## Docker enviroment
We provide a docker image for running this code
```
docker pull yaozhong/deep_intra_sv:0.9
```
* ubuntu 16.04.4
* Tensorflow 1.8.0
* Keras 2.2.4

```
WORKING_FOLD=<Absolute-PATH>
nvidia-docker run -it --rm -v $WORKING_FOLD:/workspace yaozhong/deep_intra_sv:0.9 bash
```

## Configuration file
The parameter setteing is pre-defined in the `code/config.py` file.
Some frequently used parameters can be also specifized through command line option (See '-h' option). 

### Required files
1. reference files are located in ./data/reference/, which includes required reference files: 

* reference genome FASTA file ``hs37d5.fa`` and its ``index ``	hs37d5.fa.fai``
(Please download from [1000genomes](http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/phase2_reference_assembly_sequence/),
and place it in ``./data/reference/hg19/``)
* mappability of 100-mer file
* Encode hg19 blacklist regions
* hg19 Chromesome length 

2. SV annotation files./data/SV_annotation/
  (a python script is provided to parse VCF for SV regions)

## Breakpoint enhancement for RD-based SV callers

```
# example
sample_name="simA"
binSize=400
bam_file=<path-of-bam-file>
# SV prediction of a RD-based SV caller, e.g., CNVnator
pVCF=<CNVnator-bin-resolution-prediction-file>

# trained model parameters
model=<path-of-the-trained-model>

# WGS data information, if not cached, it will re-generate.
genomeStat=<path-of-cached-BG-RD-statistics>
output_fold=<restult-saving-path>

# If gold SVs are known, it can be used for evaluating enhancement effect.
gVCF=<path-of-gold-standard-SV>

python eval/enhance_bk.py -d $sample_name -bam $bam_file -b $binSize -gs $genomeStat -mp $model -v $pVCF -o $output_fold
```

## Training UNet model

Input are WGS bam file(s) and VCF file(s). 
Cached train-test data will be first searched according to current parameters,
If cache files are not found, the code will process the bam file and cache the data.
A multi-core version of pysam is used. 
By default, all cores will be used to generate RD bins from bam file. For each training data, 
background statistics of RDs are first calcuated through sampling WGS data.
Background statistics of each WGS data will be cached in `./data/data_cache/`.

We provided a default hyperparameters of UNet and CNN in ``./experiment/model_param/``
Users can make changes of the parameter file or specifiy through command line option.
If no model parameter file is provided, the code will use hyperOpt to search preDefined hyper-parameter spaces based on the train set.

There are two evluation metrics, which are determined through CMD parameter -em (evluation mode)
* in-sample: ``-em single``
* cross-sample: ``-em cross``

For the in-sample case, only one bam file is required and the data is split into train-test with the following data split -ds option:

* Stratify: Stratified Random split
* RandRgs: Random split
* CV: cross valdiation

The test-split-proportion can be specified with option ``-tsp``
For the cross-sample case, the second bam file used as the test set is assigned through ``-d2`` option.

```
# Example
python train.py -b 400 -em single -ds Stratify -d na12878_60x -da 0 -m UNet -g 0 -mp ../experiment/model_param/unet_default
```








