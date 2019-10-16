This example demonstrates a simple deep learning model on the MNIST dataset using tf.keras.

# Train the network
First, train the network using
```bash
python train.py [--epochs=4]
```
This trains the network briefly and stores the network weights.


# Test the network
To test the network, in one terminal run
```bash
source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate
cd $HE_TRANSFORMER/examples/MNIST/MLP
python test.py --batch_size=100 \
               --encryption_parameters=$HE_TRANSFORMER/configs/he_seal_ckks_config_N11_L1.json \
               --encrypt_server_data=true
```

See the [Cryptonets-Relu example](https://github.com/NervanaSystems/he-transformer/blob/master/examples/MNIST/Cryptonets-Relu/README.md) for more details and possible configurations to try.
