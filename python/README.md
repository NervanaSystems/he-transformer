This directory contains python bindings for an experimental client.

***Note***: The client is experimental, and currently uses a large amount of memory.
For a better experience, just omit the `NGRAPH_ENABLE_CLIENT=1` flag in any examples. Then, the server will run the model without sending messages to the client.

# Building python bindings
Once you have installed he-transformer (i.e. run `make install`),
```bash
cd $HE_TRANSFORMER/build
source external/venv-tf-py3/bin/activate
make install python_client
```
This will create `python/dist/he_seal_client-*.whl`. Install it using
```bash
pip install python/dist/he_seal_client-*whl
```
To check the installation worked correctly, run
```bash
python -c "import he_seal_client"
```
This should run without errors.

# Testing python bindings
To test:
  1. In one terminal, run
  ```bash
source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate
NGRAPH_ENABLE_CLIENT=1 NGRAPH_ENCRYPT_DATA=1 NGRAPH_TF_BACKEND=HE_SEAL python $HE_TRANSFORMER/examples/ax.py
  ```

  This will start the server, which will wait for encrypted inputs from a client.

  2. In another terminal, run
  ```bash
source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate
python $HE_TRANSFORMER/examples/pyclient.py
  ```

  This will provide encrypted inputs to the HEBackend. Once the computation is complete, the output will be returned to the client and decrypted. As expected, the outputs from the server (on `ax.py`) will be inaccurate, since they are decrypted with the wrong secret key.

  The server-client approach currently works only for functions with one input parameter tensor and one result tensor.