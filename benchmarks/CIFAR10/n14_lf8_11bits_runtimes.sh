#!/bin/bash

mkdir -p exp_runtimes

cd ../../examples/CIFAR10

for i in {1..10}
do
  # CNN True True (batch size 4096) - 62.4% accuracy; 867049ms => 200ms / image
  NGRAPH_HE_SEAL_CONFIG=ckks_config_14_depth11.json NGRAPH_BATCH_DATA=1 NGRAPH_BATCH_TF=1 NGRAPH_ENCRYPT_DATA=1 NGRAPH_TF_BACKEND=HE_SEAL_CKKS python test.py --model=cnn --batch_size=4096 --batch_norm=True --train_poly_act=True --optimize_inference=True  > ../../benchmarks/CIFAR10/exp_runtimes/exp_14_11_${i}_TTT.txt
  NGRAPH_HE_SEAL_CONFIG=ckks_config_14_depth11.json NGRAPH_BATCH_DATA=1 NGRAPH_BATCH_TF=1 NGRAPH_ENCRYPT_DATA=1 NGRAPH_TF_BACKEND=HE_SEAL_CKKS python test.py --model=cnn --batch_size=4096 --batch_norm=True --train_poly_act=False --optimize_inference=True > ../../benchmarks/CIFAR10/exp_runtimes/exp_14_11_${i}_TFT.txt
done

cd -