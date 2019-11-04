#!/bin/sh
directory=$1
mode=$2

if [ "$mode" == "alt" ]
then
    dst_dir='../../baseline_results'    
    if [ ! -d $dst_dir ] 
    then
	mkdir -p $dst_dir
    fi
elif [ "$mode" == "opt" ]
then
    dst_dir='../../baseline_results'
    if [ ! -d $dst_dir ] 
    then
	mkdir -p $dst_dir
    fi
elif [ "$mode" == "par" ]
then
    dst_dir='../../baseline_results'
    if [ ! -d $dst_dir ] 
    then
	mkdir -p $dst_dir
    fi
fi
echo $dst_dir

for filename in $directory/*.txt; do
    f=$(basename -- "$filename")
    f=${f%.txt}
    python make_fsa_graph.py $filename $mode > $dst_dir/$f+FSA.txt
    echo $f is done
done
