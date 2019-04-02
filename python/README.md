This directory contains python bindings for an experimental client.

# Building python bindings
Once you have installed he-transformer (i.e. run `make install`), and with the `
```bash
make python_wheel
```
from the main build folder `$HE_TRANSFORMER/build`

This will create `python/dist/he_seal_client-*.whl`

Install using
```bash
pip install python/dist/he_seal_client-*whl
```

# Testing python bindings
To test:
  1. In one terminal, from `$HE_TRANSFORMER/build` folder, run
  ```bash
source external/venv-tf-py3/bin/activate && cd ../examples
NGRAPH_ENABLE_CLIENT=1 NGRAPH_ENCRYPT_DATA=1 NGRAPH_TF_BACKEND=HE_SEAL_CKKS python ax.py
  ```

  This will start the server, which will wait for encrypted inputs from a client.

  2. Once the server is running, in another terminal, from `$HE_TRANSFORMER/examples` folder, run
  ```bash
  python pyclient.py
  ```

  This will provide encrypted inputs to the HEBackend. Once the computation is complete, the output will be returned to the client and decrypted. As expected, the output from the server (on `ax.py`) will be nonsense.

  The server-client approach currently works only for functions with one input parameter tensor.