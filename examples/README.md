This example demonstrates a simple example of a small matrix multiplication and addition. This example depends on the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/NervanaSystems/ngraph-tf). Make sure the python environment with the ngraph-tf bridge is active, i.e. run `source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate`.

To run on the CKKS backend,
```bash
NGRAPH_TF_BACKEND=HE_SEAL_CKKS python axpy.py
```
To run on the BFV backend,
```bash
NGRAPH_TF_BACKEND=HE_SEAL_BFV python axpy.py
```

Note, the BFV encryption scheme suports only integers. For floating-point support, use the CKKS encryption scheme.

#  Client-server model
In pratice, the public key and secet should will not reside on the same object.

For a simple demonstration of a server-client approach, run
`NGRAPH_ENABLE_CLIENT=1 NGRAPH_ENCRYPT_DATA=1 NGRAPH_TF_BACKEND=HE_SEAL_CKKS python ax.py`

This will discard the Tensorflow inputs and instead wait for a client to connect and provide encrypted inputs.

To connect the client and pass inputs, in a separate terminal on the same host, run `./test/client_server/main_client`.

This will provide encrypted inputs to the HEBackend. Once the computation is complete, the output will be returned to the client and decrypted. As expected, the output from the server (on `ax.py`) will be nonsense.

The server-client approach currently works only for functions with one input parameter tensor.