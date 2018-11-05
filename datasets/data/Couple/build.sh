#!/bin/bash

fc_size=3
data_size=1000
train_size=700
test_size=300

python couple.py $fc_size $data_size > ./raw/raw.svm
cd ./raw
./split.sh $train_size $test_size

