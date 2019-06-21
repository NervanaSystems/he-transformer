This example demonstrates a simple example of a small matrix multiplication and addition. This example depends on the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/tensorflow/ngraph-bridge). Make sure the python environment with the ngraph-tf bridge is active, i.e. run `source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate`.

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

# List of command-line flags
  * `NGRAPH_ENCRYPT_DATA`. Set to 1 to encrypt data
  * `NGRAPH_ENCRYPT_MODEL`. Set to 1 to encrypt model
  * `NGRAPH_VOPS`. Set to `all` to print information about every operation performed. Set to a comma-separated list to print information about those ops; for example `NGRAPH_VOPS=add,multiply,convolution`
  * `STOP_CONST_FOLD`. Set to 1 to stop constant folding optimization. Note, this speeds up the graph compilation time for large batch sizes.
  * `NGRAPH_TF_BACKEND`. Set to `HE_SEAL` to use the HE backend. Set to `CPU` for inference on un-encrypted data
  * `NGRAPH_COMPLEX_PACK`. Set to 1 to enable complex packing. For models with no ciphertext-ciphertext multiplication, this will double the capacity from `N/2` to `N`. As a rough guideline, this flag is suitable when the model does not contain polynomial activations, and when either the model or data remains unencrypted
  * `OMP_NUM_THREADS`. Set to 1 to enable single-threaded execution (useful for debugging). For best multi-threaded performance, this number should be tuned.
  * `NGRAPH_HE_SEAL_CONFIG`. Used to specify the encryption parameters filename. If no value is passed, a small parameter choice will be used. ***Warning***: the default parameter selection does not enforce any security level. The configuration file should be of the form:
    ```bash
    {
      "scheme_name": "HE_SEAL",
      "poly_modulus_degree": 4096,
      "security_level": 128,
      "coeff_modulus": [
        30,
        22,
        22,
        30
      ]
    }
    ```
    - `scheme_name` should always be "HE_SEAL".
    - `poly_modulus_degree` should be a power of two in {1024, 2048, 4096, 8192, 16384}.
    - `security_level` should be in {0, 128, 192, 256}. Note: a security level of 0 indicates the HE backend will *not* enforce a minimum security level. This means the encryption is not secure against attacks.
    - `coeff_modulus` should be a list of integers in [1,60]. This indicates the bit-widths of the coefficient moduli used.
  * `NAIVE_RESCALING`. For comparison purposes only. No need to enable.

