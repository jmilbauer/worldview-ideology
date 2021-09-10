#!/usr/bin/bash
    
DIMS=100
WINDOW=5
SAMPLE='.00001'

python3 $(dirname $0)/train_model.py $1 $2 $DIMS $WINDOW $SAMPLE "multi"