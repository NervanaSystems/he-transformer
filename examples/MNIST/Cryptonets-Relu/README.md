This example demonstrates a simple CNN, which achieves ~98% on MNIST.
The architecture uses MaxPool and ReLU activations.

Since it is impossible to perform ReLU using homomorphic encryption, this model will only run when `NGRAPH_ENABLE_CLIENT=1`. The client will send encrypted data to the server. To perform the ReLU/Maxpool layer, the encrypted data is sent to the client, which decrypts, performs the ReLU/Maxpool, re-encrypts and sends the post-ReLU/Maxpool ciphertexts back to the server.

***Note***: the client is an experimental feature and currently uses a large amount of memory. For a better experience, see the `Debugging` section below.

This example depends on the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/tensorflow/ngraph-bridge). Make sure the python environment with ngraph-tf bridge is active, i.e. run `source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate`. Also ensure the `he_seal_client` wheel has been installed (see `python` folder for instructions).

# Train the network
First, train the network using
```bash
python train.py
```
This trains the network briefly and stores the network weights.

# Test the network
To test the network, in one terminal run
```bash
source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate
cd $HE_TRANSFORMER/examples/MNIST/MLP
```

```bash
NGRAPH_ENABLE_CLIENT=1 \
NGRAPH_ENCRYPT_DATA=1 \
NGRAPH_HE_SEAL_CONFIG=$HE_TRANSFORMER/configs/he_seal_ckks_config_N11_L1.json \
NGRAPH_TF_BACKEND=HE_SEAL \
python test.py --batch_size=1024
```
This runs inference on the Cryptonets network using the SEAL CKKS backend.
The `he_seal_ckks_config_N11_L1.json` file specifies the parameters which to run the model on. Note: the batch size must be between 1 and 1024 = 2^(11)/2.

In another terminal, run
```bash
source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate
cd $HE_TRANSFORMER/examples/MNIST
```

```bash
python pyclient_mnist.py --batch_size=1024
```

# Debugging
For debugging purposes, you can omit the use of the client.
This will perform non-linear layers on the server, which stores the public and secret keys. Note, this is not a valid security model, and is used only for debugging.

```bash
NGRAPH_ENCRYPT_DATA=1 \
NGRAPH_HE_SEAL_CONFIG=$HE_TRANSFORMER/configs/he_seal_ckks_config_N11_L1.json \
NGRAPH_TF_BACKEND=HE_SEAL \
python test.py --batch_size=1024
```

# Complex packing
For models with no ciphertext-ciphertext multiplication, use the `NGRAPH_COMPLEX_PACK=1` flag to double the capacity.
As a rough guideline, the NGRAPH_COMPLEX_PACK flag is suitable when the model does not contain polynomial activations,
and when either the model or data remains unencrypted.

Using the `NGRAPH_COMPLEX_PACK` flag, we double the capacity to 2048, doubling the throughput.

```bash
NGRAPH_COMPLEX_PACK=1 \
NGRAPH_ENCRYPT_DATA=1 \
NGRAPH_HE_SEAL_CONFIG=$HE_TRANSFORMER/configs/he_seal_ckks_config_N11_L1.json \
NGRAPH_TF_BACKEND=HE_SEAL \
python test.py --batch_size=2048
```