This example demonstrates a simple CNN, which achieves ~98% on MNIST.
The architecture uses MaxPool and ReLU activations.

Since it is impossible to perform ReLU using homomorphic encryption, this model will only run when `NGRAPH_ENABLE_CLIENT=1`. The client will send encrypted data to the server. To perform the ReLU/Maxpool layer, the encrypted data is sent to the client, which decrypts, performs the ReLU/Maxpool, re-encrypts and sends the post-ReLU/Maxpool ciphertexts back to the server.

This example depends on the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/NervanaSystems/ngraph-tf). Make sure the python environment with ngraph-tf bridge is active, i.e. run `source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate`. Also ensure the `he_seal_client` wheel has been installed (see `python` folder for instructions).

# Train the network
First, train the network using
```
python train.py
```
This trains the network briefly and stores the network weights.

# Test the network
## Python
To test the network, in one terminal run
```
source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate
cd $HE_TRANSFORMER/examples/MNIST-MLP
NGRAPH_ENABLE_CLIENT=1 NGRAPH_ENCRYPT_DATA=1 NGRAPH_BATCH_DATA=1 NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_N12_L3.json NGRAPH_TF_BACKEND=HE_SEAL python test.py --batch_size=2048 --report_accuracy=1
```
This runs inference on the Cryptonets network using the SEAL CKKS backend.
The `he_seal_ckks_config_N12_L3.json` file specifies the parameters which to run the model on. Note: the batch size must be beweteen 1 and 2048 = 2^(12)/2.

In another terminal, run
```
source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate
cd $HE_TRANSFORMER/examples
python pyclient_mnist.py --batch_size=2048
```

# Custom coefficient moduli
For improved performance at the cost of lower precision, you may select custom coefficient moduli smaller than 30 bits. This enables the selection of a smaller polynomial modulus degree N=2^11 reducing the maximum batch size to N/2=1024

For an example of this, in one terminal run
```
source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate
cd $HE_TRANSFORMER/examples/MNIST-MLP
NGRAPH_ENABLE_CLIENT=1 NGRAPH_ENCRYPT_DATA=1 NGRAPH_BATCH_DATA=1 NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_N11_L3_18bits.json NGRAPH_TF_BACKEND=HE_SEAL python test.py --batch_size=1024 --report_accuracy=1
```

In another terminal, run
```
source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate
cd $HE_TRANSFORMER/examples
python pyclient_mnist.py --batch_size=1024
```

Note: `src/get_coefficient_moduli.py` can be used to generate custom coefficient moduli

