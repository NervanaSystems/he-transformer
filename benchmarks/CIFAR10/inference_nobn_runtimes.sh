#!/bin/bash

mkdir -p no_bn_runtimes

cd ../../examples/CIFAR10

for i in {1..10}
do

  NGRAPH_HE_SEAL_CONFIG=./ckks_config_debug.json NGRAPH_ENCRYPT_DATA=1 NGRAPH_TF_BACKEND=HE_SEAL_CKKS NGRAPH_ENABLE_SERIALIZE=1 python test.py --model=cnn --batch_norm=True --train_poly_act=True --batch_size=1 --optimize_inference=False > ../../benchmarks/CIFAR10/no_bn_runtimes/exp${i}_TTF.txt
  NGRAPH_HE_SEAL_CONFIG=./ckks_config_debug.json NGRAPH_ENCRYPT_DATA=1 NGRAPH_TF_BACKEND=HE_SEAL_CKKS NGRAPH_ENABLE_SERIALIZE=1 python test.py --model=cnn --batch_norm=True --train_poly_act=False --batch_size=1 --optimize_inference=False > ../../benchmarks/CIFAR10/no_bn_runtimes/exp${i}_TFF.txt

done
cd -