# MobileNet V2 example

This folder demonstrates an example of inference on MobileNetV2.
Note: this is a work in progress, and requires ~150GB memory.
Runtime will be very slow without many cores.

See here: https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
for a description. We use mobilenet_v2_0.35_96 and resize images to `84x84x3`

# To peform inference
1. Make sure python env is active, i.e. run
```bash
cd $HE_TRANSFORMER/build
source external/venv-tf-py3/bin/activat
```
Also be sure the `he_seal_client` wheel has been installed

2. Add TensorFlow graph transforms to your path, i.e. run
```bash
export PATH=$HE_TRANSFORMER/build/ext_ngraph_tf/src/ext_ngraph_tf/build_cmake/tensorflow/bazel-bin/tensorflow/tools/graph_transforms:$PATH
```

2. To download the models and optimize for inference, call
```bash
python get_models.py
```

3. To enable image processing, run
```bash
pip install pillow
```

4. To perform inference
    1. In one terminal, run
      ```bash
      NGRAPH_ENABLE_CLIENT=1 \
      NGRAPH_ENCRYPT_DATA=1 \
      NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_N12_L4.json \
      NGRAPH_TF_BACKEND=HE_SEAL \
      python test.py
      ```

    2. In a second terminal on the same server (with the python env active), run
    ```bash
    python client.py
    ```
Upon successful completion, the client will output the top5 categories for an image of Grace Hopper in a military uniform:
```
top5 [922 458 721 653 835]
['book jacket' 'bow tie' 'pill bottle' 'military uniform' 'suit']
```

# Debugging
1. For debugging purposes, run the model in plaintext with
```bash
NGRAPH_TF_BACKEND=HE_SEAL python test.py
```

2. To run the model without the client, (encryption and decryption will occur locally, so this isn't privacy-preserving):
```bash
NGRAPH_ENCRYPT_DATA=1 \
NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_N12_L4.json \
NGRAPH_TF_BACKEND=HE_SEAL \
python test.py
```

# Fastest
For faster runtime, try
```bash
OMP_NUM_THREADS=56 \
NGRAPH_ENCRYPT_DATA=1 \
NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_N12_L4.json \
NGRAPH_TF_BACKEND=HE_SEAL \
python test.py
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

4. To run inference using TensorFlow on unencrypted data, call
```bash
python test.py \
  --data_dir=$DATA_DIR
  --batch_size=300
```

5. To call inference using HE_SEAL's plaintext operations (for debugging), call
```bash
NGRAPH_TF_BACKEND=HE_SEAL \
STOP_CONST_FOLD=1 \
python test.py \
--data_dir=$DATA_DIR \
--ngraph=true \
--batch_size=300
```
Note: this will result in many outputs. To suppress these, pass the `NGRAPH_SILENCE_OPS=1` flag

6. To call inference using encrypted data, run the below command. ***Warning***: this will take ~210GB memory.
```bash
OMP_NUM_THREADS=56 \
STOP_CONST_FOLD=1 \
NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_N12_L4.json \
NGRAPH_TF_BACKEND=HE_SEAL \
NGRAPH_ENCRYPT_DATA=1 \
python test.py \
--data_dir=$DATA_DIR \
--ngraph=true \
--batch_size=2048
```

7. To double the throughput using complex packing, run:
```bash
OMP_NUM_THREADS=56 \
STOP_CONST_FOLD=1 \
NGRAPH_COMPLEX_PACK=1 \
NGRAPH_TF_BACKEND=HE_SEAL \
NGRAPH_ENCRYPT_DATA=1 \
NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_N12_L4.json NGRAPH_BATCH_DATA=1 \
python test.py \
--data_dir=$DATA_DIR \
--ngraph=true \
--batch_size=4096