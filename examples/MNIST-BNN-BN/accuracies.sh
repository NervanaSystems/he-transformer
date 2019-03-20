#!/bin/bash

mkdir -p accuracies

# For some reason, normal piping of stdout doesn't capture the tensorflow output,

for i in {1..10}
do
sh -c 'python train.py --train_loop_count=100000' 2>&1 | tee accuracies/acc_${i}.txt
done