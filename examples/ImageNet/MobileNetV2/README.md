# MobileNet V2 example

This folder demonstrates an example of inference on MobileNetV2.
Note: this is a work in progress, and requires ~50GB memory.
Runtime will be very slow without many cores.

See https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
for a description.

# Setup
1. Make sure python env is active, i.e. run
```bash
source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate
```
Also ensure the `pyhe_client` wheel has been installed (see `python` folder for instructions).

The examples rely on numpy and pillow, so run
```bash
pip install numpy pillow
```

2. Build Tensorflow graph transforms and add them to your path:

To build run:
```bash
cd $HE_TRANSFORMER/build/ext_ngraph_tf/src/ext_ngraph_tf/build_cmake/tensorflow
bazel build tensorflow/tools/graph_transforms:transform_graph
```

To add to path run:
```bash
export PATH=$HE_TRANSFORMER/build/ext_ngraph_tf/src/ext_ngraph_tf/build_cmake/tensorflow/bazel-bin/tensorflow/tools/graph_transforms:$PATH
```

3. To download the models and optimize for inference, call
```bash
python get_models.py
```

# Image-Net evaluation
1. First, sign up for an account at image-net.org
2. Download the 2012 test_images (all tasks)) 13GB MD5: `e64ceb247e473635708aed23ab6d839` file on image-net.org

Extract the validation images:
```bash
tar -xf ILSVRC2012_img_test.tar
```
3. Download development kit (Task 1 & 2) and extract `validation_ground_truth.txt`

The directory setup should be:
```
DATA_DIR/validation_images/ILSVRC2012_val_00000001.JPEG
DATA_DIR/validation_images/ILSVRC2012_val_00000002.JPEG
...
DATA_DIR/validation_images/ILSVRC2012_val_00050000.JPEG
DATA_DIR/ILSVRC2012_validation_ground_truth.txt
```
for some `DATA_DIR` folder.

For the remaining instructions, run```bash
export DATA_DIR=path_to_your_data_dir
```

## CPU backend
To run inference using the CPU backend on unencrypted data, call
```bash
python test.py \
  --data_dir=$DATA_DIR \
  --batch_size=300 \
  --backend=CPU
```

5. To call inference using HE_SEAL's plaintext operations (for debugging), call
```bash
STOP_CONST_FOLD=1 \
python test.py \
--data_dir=$DATA_DIR \
--batch_size=300 \
--backend=HE_SEAL
```
Note, the `STOP_CONST_FOLD` flag will prevent the constant folding graph optimization.
For large batch sizes, const folding incurs significant overhead during graph compilation, and doesn't result in much runtime speedup.

  5.a To try on a larger model, call:
  ```bash
  STOP_CONST_FOLD=1 \
  NGRAPH_VOPS=all \
  NGRAPH_TF_BACKEND=HE_SEAL \
  python test.py \
  --image_size=128 \
  --data_dir=$DATA_DIR \
  --batch_size=30 \
  --model=./model/mobilenet_v2_0.35_128_opt.pb \
  --ngraph=true
  ```

6. To call inference using encrypted data, run the below command. ***Warning***: this will take ~50GB memory.
```bash
OMP_NUM_THREADS=56 \
STOP_CONST_FOLD=1 \
NGRAPH_HE_SEAL_CONFIG=$HE_TRANSFORMER/configs/he_seal_ckks_config_N12_L4.json \
NGRAPH_TF_BACKEND=HE_SEAL \
NGRAPH_ENCRYPT_DATA=1 \
python test.py \
--data_dir=$DATA_DIR \
--ngraph=true \
--batch_size=2048
```

6a. To try on a larger model, call:
  ```bash
  STOP_CONST_FOLD=1 \
  OMP_NUM_THREADS=56 \
  NGRAPH_TF_BACKEND=HE_SEAL \
  NGRAPH_HE_SEAL_CONFIG=$HE_TRANSFORMER/configs/he_seal_ckks_config_N12_L4.json \
  NGRAPH_ENCRYPT_DATA=1 \
  python test.py \
  --image_size=128 \
  --data_dir=$DATA_DIR \
  --ngraph=true \
  --model=./model/mobilenet_v2_0.35_128_opt.pb \
  --batch_size=30
  ```

7. To double the throughput using complex packing, run the below command.  ***Warning***: this will take ~120GB memory.
```bash
OMP_NUM_THREADS=56 \
STOP_CONST_FOLD=1 \
NGRAPH_COMPLEX_PACK=1 \
NGRAPH_TF_BACKEND=HE_SEAL \
NGRAPH_ENCRYPT_DATA=1 \
NGRAPH_HE_SEAL_CONFIG=$HE_TRANSFORMER/configs/he_seal_ckks_config_N12_L4.json \
python test.py \
--data_dir=$DATA_DIR \
--ngraph=true \
--batch_size=4096

8. To enable the client, in one terminal, run:
```bash
NGRAPH_ENABLE_CLIENT=1 \
OMP_NUM_THREADS=56 \
STOP_CONST_FOLD=1 \
NGRAPH_VOPS=BoundedRelu \
NGRAPH_COMPLEX_PACK=1 \
NGRAPH_ENCRYPT_DATA=1 \
NGRAPH_TF_BACKEND=HE_SEAL \
NGRAPH_HE_SEAL_CONFIG=$HE_TRANSFORMER/configs/he_seal_ckks_config_N12_L4.json \
python test.py \
  --batch_size=4096  \
  --image_size=96 \
  --ngraph=true \
  --model=./model/mobilenet_v2_0.35_96_opt.pb \
  --data_dir=$DATA_DIR \
  --ngraph=true
```
Since this will take a while to run, you may want to add verbosity, e.g.
the `NGRAPH_VOPS=all` flag, to the above command.

In another terminal (with the python environment active), run
```bash
OMP_NUM_THREADS=56 \
NGRAPH_COMPLEX_PACK=1 \
python client.py \
  --batch_size=4096 \
  --image_size=96 \
  --data_dir=$DATA_DIR
```