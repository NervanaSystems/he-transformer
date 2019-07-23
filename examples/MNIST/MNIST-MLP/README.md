# Train the network
First, train the network using
```bash
python train.py [--train_loop_count=20000]
```
This trains the network briefly and stores the network weights.


# Test the network
To test the network, in one terminal run
```bash
source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate
cd $HE_TRANSFORMER/examples/MNIST/MNIST-MLP
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