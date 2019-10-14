This example demonstrates the [Cryptonets](https://www.microsoft.com/en-us/research/publication/cryptonets-applying-neural-networks-to-encrypted-data-with-high-throughput-and-accuracy/) network, which achieves ~99% accuracy on MNIST.

This example depends on the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/tensorflow/ngraph-bridge). Make sure the python environment with ngraph-tf bridge is active, i.e. run `source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate`.

# Train the network
First, train the network using
```
python train.py
```
This trains the network briefly and stores the network weights.

# Test the network
To test the network, with encrypted data,
```
python test.py --batch_size=4096 \
               --encryption_parameters=$HE_TRANSFORMER/configs/he_seal_ckks_config_N13_L7.json \
               --encrypt_server_data=True
```

This runs inference on the Cryptonets network with encrypted data using the SEAL CKKS backend.
See the [Cryptonets-Relu example](https://github.com/NervanaSystems/he-transformer/blob/master/examples/MNIST/Cryptonets-Relu/README.md) for more details and possible configurations to try.
