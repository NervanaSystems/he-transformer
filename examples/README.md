This example demonstrates a simple example of a small matrix multiplication and addition. This example depends on the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/tensorflow/ngraph-bridge). Make sure the python environment with the ngraph-tf bridge is active, i.e. run `source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate`.

The examples rely on numpy, so first run
```bash
pip install numpy
```

To run on the CPU backend,
```bash
python $HE_TRANSFORMER/examples/ax.py --backend=CPU
```

To run on the CKKS backend,
```bash
python $HE_TRANSFORMER/examples/ax.py --backend=HE_SEAL
```

By default, the default encryption parameters will be used. To specify a non-default set of parameters, use the `encryption_parameters` flag, for example
```bash
python $HE_TRANSFORMER/examples/ax.py --backend=HE_SEAL --encryption_parameters=$HE_TRANSFORMER/configs/he_seal_ckks_config_N11_L1.json
 ```

#  Client-server model
In practice, the public key and secret key will not be stored in the same location.
Instead, a client will provide the backend with encrypted data.

The client uses python bindings. See the `python` folder for instructions to build he-transformer with python bindings.

For a simple demonstration of a server-client approach, run
```bash
python $HE_TRANSFORMER/examples/ax.py --backend=HE_SEAL --enable_client=yes
```

This will discard the Tensorflow inputs and instead wait for a client to connect and provide encrypted inputs.
To start the client, in a separate terminal on the same host (with the ngraph-tf bridge python environment active), run
```bash
python $HE_TRANSFORMER/examples/pyclient.py
```

Once the computation is complete, the output will be returned to the client and decrypted. The server will attempt decrypt the output as well; however, since it does not have the client's secret key, the output will be meaningless.

The server-client approach currently works only for functions with one result tensor.

For a deep learning example using the client-server model, see the `MNIST/MLP` folder.

# List of command-line flags
  * `STOP_CONST_FOLD`. Set to 1 to stop constant folding optimization. Note, this speeds up the graph compilation time for large batch sizes.
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
      ],
      "scale": 4194304,
      "complex_packing": true,
    }
    ```
    - `scheme_name` should always be "HE_SEAL".
    - `poly_modulus_degree` should be a power of two in {1024, 2048, 4096, 8192, 16384}.
    - `security_level` should be in {0, 128, 192, 256}. Note: a security level of 0 indicates the HE backend will *not* enforce a minimum security level. This means the encryption is not secure against attacks.
    - `coeff_modulus` should be a list of integers in [1,60]. This indicates the bit-widths of the coefficient moduli used. ***Note***: The number of coefficient moduli should be at least the multiplicative depth of your model between non-polynomial layers.
    - `scale` is the scale at which number are encoded; `log2(scale)` represents roughly the fixed-bit precision of the encoding. If no scale is passes, the second-to-last coeffcient modulus is used.
    - `complex_packing` specifies whether or not to double the capacity (i.e. maximum batch size) by packing two scalars `(a,b)` in a complex number `a+bi`. Typically, the capacity is `poly_modulus_degree/2`. Enabling complex packing doubles the capacity to `poly_modulus_degree`. Note: enabling `complex_packing` will reduce the performance of ciphertext-ciphertext multiplication.
  * `NGRAPH_HE_VERBOSE_OPS`. Set to `all` to print information about every operation performed. Set to a comma-separated list to print information about those ops; for example `NGRAPH_HE_VERBOSE_OPS=add,multiply,convolution`. *Note*, `NGRAPH_HE_LOG_LEVEL` should be set to at least 3 when using `NGRAPH_HE_VERBOSE_OPS`
  * `NGRAPH_HE_LOG_LEVEL`. Defines the verbosity of the logging. Set to 0 for minimal logging, 5 for maximum logging. Roughly:
    - `NGRAPH_HE_LOG_LEVEL=0 [default]` will print minimal amount of information
    - `NGRAPH_HE_LOG_LEVEL=1` will print encryption parameters
    - `NGRAPH_HE_LOG_LEVEL=3` will print op information (when `NGRAPH_HE_VERBOSE_OPS` is enabled)
    - `NGRAPH_HE_LOG_LEVEL=4` will print communication information
    - `NGARPH_HE_LOG_LEVEL=5` is the highest debug level

  # Creating your own DL model
  We currently only support DL models with a single `Parameter`, as is the case for most standard DL models. During training, the weights may be TensorFlow `Variable` ops, which translate to nGraph `Parameter` ops. In this case, he-transformer will be unable to tell what tensor represents the data to encrypt. So, you will need to convert the ops representing the model weights to `Constant` ops. TensorFlow, for example, has a `freeze_graph` utility to do so. See the `MNIST/MLP` folder for an example using `freeze_graph`.