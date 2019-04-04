This directory contains python bindings for an experimental client.

# Building python bindings
Once you have installed he-transformer (i.e. run `make install`),
```bash
cd $HE_TRANSFORMER/build
source external/venv-tf-py3/bin/activate
make python_wheel
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
  cd $HE_TRANSFORMER/build
source external/venv-tf-py3/bin/activate && cd ../examples
NGRAPH_ENABLE_CLIENT=1 NGRAPH_ENCRYPT_DATA=1 NGRAPH_TF_BACKEND=HE_SEAL_CKKS python ax.py
  ```

  This will start the server, which will wait for encrypted inputs from a client.

  2. Once the server is running, in another terminal run
  ```bash
  cd $HE_TRANSFORMER/examples
  python pyclient.py
  ```

  This will provide encrypted inputs to the HEBackend. Once the computation is complete, the output will be returned to the client and decrypted. As expected, the output from the server (on `ax.py`) will be nonsense.

  The server-client approach currently works only for functions with one input parameter tensor.