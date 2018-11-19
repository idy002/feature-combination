#!/bin/bash

data_size=10000
train_size=7000
test_size=3000

python3 couple.py $data_size > ./raw/raw.svm
cd ./raw
./split.sh $train_size $test_size

