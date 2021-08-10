#!/bin/bash

dataset=$1
output=$2

mkdir -p $output/{train,test}

for cls in `ls $dataset`
do
    for subset in train test
    do
	mkdir -p $output/$subset/$cls
	cd $output/$subset/$cls
	
	for f in `ls ../../../$dataset/$cls/$subset/`
	do
	    ln -s ../../../$dataset/$cls/$subset/$f .
	done
	cd ../../..
    done
done

cd $output/
ln -s test val
cd ..

rm -f $output/{train,test}/*/*.off
if [ $3 == 2 ]
then
    for ((i=0;i<20;i++))
    do
	for ((j=2;j<5;j++))
	do
	    n=`expr $i \* 4 + $j`
	    fn=`printf "%03d" $n`
	    rm -f $output/{train,test}/*/*_$fn.png
	done
    done
fi
