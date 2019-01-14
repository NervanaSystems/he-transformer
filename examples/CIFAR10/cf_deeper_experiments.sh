#!/bin/bash

MAX_STEPS=10000

mkdir -p deeper_exp
for i in {1..10}
do
  echo "Trial $i TT"
  # True, True
  python train.py --model=deeper_cnn --batch_norm=True --train_poly_act=True --max_steps=$MAX_STEPS
  python test.py --model=deeper_cnn --batch_norm=True --train_poly_act=True --batch_size=10000 > deeper_exp/cnn_bn_train_poly_exp_${i}.txt

  # True, False
  echo "Trial $i TF"
  python train.py --model=deeper_cnn --batch_norm=True --train_poly_act=False --max_steps=$MAX_STEPS
  python test.py --model=deeper_cnn --batch_norm=True --train_poly_act=False --batch_size=10000 > deeper_exp/cnn_bn_exp_${i}.txt

  # False, True
  echo "Trial $i FT"
  python train.py --model=deeper_cnn --batch_norm=False --train_poly_act=True --max_steps=$MAX_STEPS
  python test.py --model=deeper_cnn --batch_norm=False --train_poly_act=True --batch_size=10000 > deeper_exp/cnn_train_poly_exp_${i}.txt

  # False, False
  echo "Trial $i FF"
  python train.py --model=deeper_cnn --batch_norm=False --train_poly_act=False --max_steps=$MAX_STEPS
  python test.py --model=deeper_cnn --batch_norm=False --train_poly_act=False --batch_size=10000 > deeper_exp/cnn_exp_${i}.txt

done
