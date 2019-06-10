This example demonstrates a simple example of a small matrix multiplication and addition. This example depends on the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/NervanaSystems/ngraph-tf). Make sure the python environment with the ngraph-tf bridge is active, i.e. run `source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate`.

To run on the CKKS backend,
```bash
NGRAPH_TF_BACKEND=HE_SEAL python axpy.py
```

#  Client-server model
In pratice, the public key and secret key will not reside on the same object.
Instead, a client will provide the server with encrypted data.

The client uses python bindings. See the `python` folder for instructions to build he-transformer with python bindings.

For a simple demonstration of a server-client approach, run
`NGRAPH_ENABLE_CLIENT=1 NGRAPH_ENCRYPT_DATA=1 NGRAPH_TF_BACKEND=HE_SEAL python ax.py`

This will discard the Tensorflow inputs and instead wait for a client to connect and provide encrypted inputs.

To start the client, in a separate terminal on the same host, run `python pyclient.py`.

Once the computation is complete, the output will be returned to the client and decrypted. The server will attempt decrypt the output as well; however, since it does not have the client's secret key, the output will be meaningless.

The server-client approach currently works only for functions with one input parameter tensor.

For a deep learning example using the client-server model, see the `MNIST-MLP` folder.
