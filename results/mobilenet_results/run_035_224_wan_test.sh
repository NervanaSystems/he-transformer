#!/bin/bash

# Make sure the python environment is active
mkdir -p results

curr_dir=$(pwd)
echo $curr_dir

cd ../../examples/MobileNetV2

# Best performance
for i in {1..1}
do
  outfile=$curr_dir/results/035_224_wan_test.txt
  echo "Trial ${i}"
  rm -rf $outfile
  touch $outfile

  (NGRAPH_ENABLE_CLIENT=1 \
  OMP_NUM_THREADS=56 \
  STOP_CONST_FOLD=1 \
  NGRAPH_VOPS=all \
  NGRAPH_COMPLEX_PACK=1 \
  NGRAPH_ENCRYPT_DATA=1 \
  NGRAPH_TF_BACKEND=HE_SEAL \
  NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_N12_L4.json \
  python test.py \
  --batch_size=4096  \
  --image_size=224 \
  --ngraph=true \
  --model=./model/mobilenet_v2_0.35_224_opt.pb \
  --data_dir=$DATA_DIR \
  --ngraph=true &) >> $outfile


  ssh skx113 << EOF
    export HE_TRANSFORMER=/nfs/site/home/fboemer/repos/he-transformer/;
    source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate;
    outfile=$HE_TRANSFORMER/results/mobilenet_results/035_224_wan_test.txt;
    cd $HE_TRANSFORMER/examples/MobileNetV2/;
    OMP_NUM_THREADS=56 \
    NGRAPH_COMPLEX_PACK=1 \
    python client.py \
    --batch_size=4096 \
    --image_size=224 \
    --data_dir=$DATA_DIR \
    --hostname=skx114 >> $outfile;
EOF

done
cd -