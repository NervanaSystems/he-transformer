#!/bin/bash

# Make sure the python environment is active
mkdir -p results

curr_dir=$(pwd)
echo $curr_dir

cd ../../examples/MobileNetV2

# Best performance
outfile=$curr_dir/results/035_192_comms.txt
rm -rf $outfile
touch $outfile
start_batch=0
echo "Start batch" $start_batch
batch_size=40
echo "Batch size" $batch_size

(NGRAPH_ENABLE_CLIENT=1 \
OMP_NUM_THREADS=56 \
STOP_CONST_FOLD=1 \
NGRAPH_VOPS=all \
NGRAPH_COMPLEX_PACK=1 \
NGRAPH_ENCRYPT_DATA=1 \
NGRAPH_TF_BACKEND=HE_SEAL \
NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_N12_L4.json \
python test.py \
--batch_size=$batch_size  \
--image_size=192 \
--ngraph=true \
--start_batch=$start_batch \
--model=./model/mobilenet_v2_0.35_192_opt.pb \
--data_dir=$DATA_DIR \
--ngraph=true &) >> $outfile

# Let server start
sleep 15s
OMP_NUM_THREADS=56 \
NGRAPH_COMPLEX_PACK=1 \
python client.py \
--batch_size=$batch_size  \
--start_batch=$start_batch \
--image_size=192 \
--data_dir=$DATA_DIR >> $outfile

cd -

# fa23068e327c74156ca17a7e075f18976c95e867 WIP for more accuracte mobilenet results
# 15c192315b3803acaf2ab148a1eed6a60b04df7c works July 22