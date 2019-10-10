#!/bin/bash
data_dir=$1
model=$2
echo Downloading $model
wget -P $data_dir http://dl.fbaipublicfiles.com/fairseq/models/spanbert_$model.tar.gz
mkdir $data_dir/$model
tar xvzf $data_dir/spanbert_$model.tar.gz -C $data_dir/$model
rm $data_dir/spanbert_$model.tar.gz
