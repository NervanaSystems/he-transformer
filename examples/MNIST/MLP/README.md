This example demonstrates a simple deep learning model on the MNIST dataset. It demonstrates how to save a model in protobuf format with frozen weights.

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
cd $HE_TRANSFORMER/examples/MNIST/MLP
NGRAPH_HE_SEAL_CONFIG=$HE_TRANSFORMER/configs/he_seal_ckks_config_N11_L1.json \
python test.py --enable_client=yes
```
This runs inference on the Cryptonets network using the SEAL CKKS backend.
The `he_seal_ckks_config_N11_L1.json` file specifies the parameters which to run the model on. Note: the batch size must be between 1 and 1024 = 2^(11)/2.

In another terminal, run
```bash
source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate
cd $HE_TRANSFORMER/examples/MNIST
python pyclient_mnist.py --batch_size=1024
```

See the [Cryptonets-Relu example](https://github.com/NervanaSystems/he-transformer/blob/master/examples/MNIST/Cryptonets-Relu/README.md) for more details and possible configurations to try.

