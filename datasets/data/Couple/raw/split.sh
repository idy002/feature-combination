#!/bin/bash

shuf raw.svm > raw.shuffled.svm
head -n $1 raw.shuffled.svm > raw.train.svm
tail -n $2 raw.shuffled.svm > raw.test.svm

