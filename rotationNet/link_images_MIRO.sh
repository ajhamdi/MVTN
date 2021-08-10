#!/bin/bash

dataset=$1
output=$2

mkdir -p $output/{train,val}

for cls in `ls $dataset`
do
    echo $cls

    mkdir -p $output/val/$cls
    cd $output/val/$cls
    for f in `ls ../../../$dataset/$cls/${cls}_1_*.png`
    do
	ln -s $f .
    done
    cd ../../..
	
    mkdir -p $output/train/$cls
    cd $output/train/$cls
    for f in `ls ../../../$dataset/$cls/*.png | grep -v "${cls}_1_"`
    do
	ln -s $f .
    done
    cd ../../..
done
