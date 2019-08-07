This folder demonstrates several examples of simple CNNs on the MNIST dataset:
  * The CryptoNets folder implements the [Cryptonets](https://www.microsoft.com/en-us/research/publication/cryptonets-applying-neural-networks-to-encrypted-data-with-high-throughput-and-accuracy/) network, which achieves ~99% accuracy on MNIST.

  * The Cryptonets-Relu folder adapts the CryptoNets network to use Relu activations instead of `x^2` activations

  * The MLP folder demonstrates saving/loading TensorFlow models.


It is impossible to perform ReLU and Maxpool using homomorphic encryption. We support these functions in two ways:

  1) A client-server model, enabled with `NGRAPH_ENABLE_CLIENT=1`. The client will send encrypted data to the server. To perform the ReLU/Maxpool layer, the encrypted data is sent to the client, which decrypts, performs the ReLU/Maxpool, re-encrypts and sends the post-ReLU/Maxpool ciphertexts back to the server.

***Note***: the client is an experimental feature and currently uses a large amount of memory. For a better experience, see the `Debugging` section below.

  2) A debugging interface (active by default). This runs ReLu/Maxpool locally.
  ***Warning***: This is not privacy-preserving, and should be used for debugging only.

These examples depends on the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/tensorflow/ngraph-bridge). Make sure the python environment with ngraph-tf bridge is active, i.e. run `source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate`. Also ensure the `he_seal_client` wheel has been installed (see `python` folder for instructions).
