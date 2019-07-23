This example demonstrates the [Cryptonets](https://www.microsoft.com/en-us/research/publication/cryptonets-applying-neural-networks-to-encrypted-data-with-high-throughput-and-accuracy/) network, which achieves ~99% accuracy on MNIST.

This example depends on the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/tensorflow/ngraph-bridge). Make sure the python environment with ngraph-tf bridge is active, i.e. run `source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate`.

# Train the network
First, train the network using
```
python train.py
```
This trains the network briefly and stores the network weights.

# Test the network
## Python
To test the network, with
  * encrypted data,
```
NGRAPH_ENCRYPT_DATA=1 \
NGRAPH_HE_SEAL_CONFIG=$HE_TRANSFORMER/configs/he_seal_ckks_config_N13_L7.json \
NGRAPH_TF_BACKEND=HE_SEAL \
python test.py --batch_size=4096
```

  * encrypted model,
```
NGRAPH_ENCRYPT_MODEL=1 \
NGRAPH_HE_SEAL_CONFIG=$HE_TRANSFORMER/configs/he_seal_ckks_config_N13_L7.json \
NGRAPH_TF_BACKEND=HE_SEAL \
python test.py --batch_size=4096
```

This runs inference on the Cryptonets network using the SEAL CKKS backend.
The `he_seal_ckks_config_N13_L7.json` file specifies the parameters which to run the model on. Note: the batch size must be beweteen 1 and 4096.
