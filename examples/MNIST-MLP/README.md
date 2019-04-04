This example demonstrates a simple CNN, which achieves ~99% on MNIST.
The architecture is identical to the [Cryptonets](https://www.microsoft.com/en-us/research/publication/cryptonets-applying-neural-networks-to-encrypted-data-with-high-throughput-and-accuracy/) network, which the exception of ReLU activations instead of square activations.

Because it is impossible to perform ReLU using homomorphic encryption, this model will only run when `NGRAPH_ENABLE_CLIENT=1`. The client will send encrypted data to the server. To perform the ReLU layer, the encrypted data is sent to the client, which decrypts, performs the ReLU, re-encrypts and sends the post-ReLU ciphertexts back to the server.

This example depends on the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/NervanaSystems/ngraph-tf). Make sure the python environment with ngraph-tf bridge is active, i.e. run `source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate`.

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
NGRAPH_ENABLE_CLIENT=1 NGRAPH_ENCRYPT_DATA=1 NGRAPH_BATCH_DATA=1 NGRAPH_BATCH_TF=1 NGRAPH_HE_SEAL_CONFIG=../../test/model/he_seal_ckks_config_12_L3.json NGRAPH_TF_BACKEND=HE_SEAL_CKKS python test.py --batch_size=2048 --report_accuracy=1
```
This runs inference on the Cryptonets network using the SEAL CKKS backend.
The `he_seal_ckks_config_12_L3.json` file specifies the parameters which to run the model on. Note: the batch size must be beweteen 1 and 2048.

In another terminal, from the `examples` folder, run
```
python pyclient_mnist.py --batch_size=2048
```