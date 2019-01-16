#!/bin/bash

mkdir -p exp
for i in {1..10}
do
  echo "Trial $i TT"
  # True, True
  python train.py --model=cnn --batch_norm=True --train_poly_act=True --max_steps=10000
  python test.py --model=cnn --batch_norm=True --train_poly_act=True --batch_size=10000 > exp/cnn_bn_train_poly_exp_${i}.txt

  # True, False
  echo "Trial $i TF"
  python train.py --model=cnn --batch_norm=True --train_poly_act=False --max_steps=10000
  python test.py --model=cnn --batch_norm=True --train_poly_act=False --batch_size=10000 > exp/cnn_bn_exp_${i}.txt

  # False, True
  echo "Trial $i FT"
  python train.py --model=cnn --batch_norm=False --train_poly_act=True --max_steps=10000
  python test.py --model=cnn --batch_norm=False --train_poly_act=True --batch_size=10000 > exp/cnn_train_poly_exp_${i}.txt

  # False, False
  echo "Trial $i FF"
  python train.py --model=cnn --batch_norm=False --train_poly_act=False --max_steps=10000
  python test.py --model=cnn --batch_norm=False --train_poly_act=False --batch_size=10000 > exp/cnn_exp_${i}.txt

done
