#!/usr/bin/bash
    
DIMS=100
WINDOW=5
SAMPLE='.00001'

for src in $1/*; do
    target="$2/$(basename $src .txt).model"
    python3 $(dirname $0)/train_model.py $src $target $DIMS $WINDOW $SAMPLE
done