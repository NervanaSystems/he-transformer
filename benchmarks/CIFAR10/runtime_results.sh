#!/bin/bash

mkdir -p exp_runtimes

for i in {1..10}
do

  # CNN True True (batch size 4096) - 62.4% accuracy; 867049ms => 200ms / image
  NGRAPH_HE_SEAL_CONFIG=ckks_config_14_depth10.json NGRAPH_BATCH_DATA=1 NGRAPH_BATCH_TF=1 NGRAPH_ENCRYPT_DATA=1 NGRAPH_TF_BACKEND=HE_SEAL_CKKS python test.py --model=cnn --batch_size=4096 --batch_norm=True --train_poly_act=True > exp_runtimes/exp_${i}_TT.txt
  NGRAPH_HE_SEAL_CONFIG=ckks_config_14_depth10.json NGRAPH_BATCH_DATA=1 NGRAPH_BATCH_TF=1 NGRAPH_ENCRYPT_DATA=1 NGRAPH_TF_BACKEND=HE_SEAL_CKKS python test.py --model=cnn --batch_size=4096 --batch_norm=True --train_poly_act=False > exp_runtimes/exp_${i}_TF.txt
  NGRAPH_HE_SEAL_CONFIG=ckks_config_14_depth10.json NGRAPH_BATCH_DATA=1 NGRAPH_BATCH_TF=1 NGRAPH_ENCRYPT_DATA=1 NGRAPH_TF_BACKEND=HE_SEAL_CKKS python test.py --model=cnn --batch_size=4096 --batch_norm=False --train_poly_act=True > exp_runtimes/exp_${i}_FT.txt
  NGRAPH_HE_SEAL_CONFIG=ckks_config_14_depth10.json NGRAPH_BATCH_DATA=1 NGRAPH_BATCH_TF=1 NGRAPH_ENCRYPT_DATA=1 NGRAPH_TF_BACKEND=HE_SEAL_CKKS python test.py --model=cnn --batch_size=4096 --batch_norm=False --train_poly_act=False > exp_runtimes/exp_${i}_FF.txt

done
