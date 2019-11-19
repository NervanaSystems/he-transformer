This example demonstrates a simple CNN, which achieves ~98% on MNIST.
The architecture uses MaxPool and ReLU activations.

Since it is not possible to date to perform ReLU using the CKKS homomorphic encryption, this model will only run with the help of a client. The client will send encrypted data to the server. To perform the ReLU/Maxpool layer, the encrypted data is sent to the client, which decrypts, performs the ReLU/Maxpool, re-encrypts and sends the post-ReLU/Maxpool ciphertexts back to the server.

This example depends on the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/tensorflow/ngraph-bridge). Make sure the python environment with ngraph-tf bridge is active, i.e. run `source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate`. Also ensure the `pyhe_client` wheel has been installed (see `python` folder for instructions).

# Train the network
First, train the network using
```bash
python train.py
```
This trains the network briefly and stores the network weights.

# Test the network
First, make sure the python virtual environment is active:
```bash
source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate
cd $HE_TRANSFORMER/examples/MNIST/Cryptonets-Relu
```

## CPU
To test the network using the CPU backend, run
```bash
python test.py --batch_size=512 --backend=CPU
```

## HE_SEAL plaintext
To test the network using plaintext inputs (i.e. not encrypted), run
```bash
python test.py --batch_size=512 --backend=HE_SEAL
```
This should just be used for debugging, since the data is not encrypted

## HE_SEAL encrypted
To test the network using encrypted inputs, run
```bash
python test.py --batch_size=1024 \
               --backend=HE_SEAL \
               --encrypt_server_data=yes \
               --encryption_parameters=$HE_TRANSFORMER/configs/he_seal_ckks_config_N11_L1.json
```

This runs inference on the Cryptonets network using the SEAL CKKS backend. Note, the client is *not* enabled, meaning the backend holds the secret and public keys. This should only be used for debugging, as it is *not* cryptographically secure.

See the [examples](https://github.com/NervanaSystems/he-transformer/blob/master/examples/README.md) for more details on the encryption parameters.

## HE_SEAL client
To test the network with inputs from a client, first install the [python client](https://github.com/NervanaSystems/he-transformer/tree/master/python). Then, in one terminal, run
```bash
source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate
cd $HE_TRANSFORMER/examples/MNIST/Cryptonets-Relu
python test.py --enable_client=yes \
               --backend=HE_SEAL \
               --encryption_parameters=$HE_TRANSFORMER/configs/he_seal_ckks_config_N11_L1.json
```

In another terminal, run
```bash
source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate
cd $HE_TRANSFORMER/examples/MNIST
python pyclient_mnist.py --batch_size=1024 \
                         --encrypt_data=yes
```

## Debugging
For debugging purposes, enable the `NGRAPH_HE_LOG_LEVEL` or `NGRAPH_HE_VERBOSE_OPS` flags. See [here](https://github.com/NervanaSystems/he-transformer/blob/master/examples/README.md) for more details.
