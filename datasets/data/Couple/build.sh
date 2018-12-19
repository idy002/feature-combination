#!/bin/bash

data_size=100000
train_size=70000
test_size=30000

python3 couple.py $data_size > ./raw/raw.svm
cd ./raw
./split.sh $train_size $test_size

