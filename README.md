# HE Transformer for nGraph

HE transformer is a Homomorphic Encryption (HE) backend to nGraph.
This is meant as a Proof-of-concept project to demonstrate the feasibility of homomorphic encryption (HE) on **local** machines.

The goal is to measure performance of various HE schemes for deep learning.

This is  **not** intended to be a production-ready product, but rather a research tool.

Currently, we support two backends, which are installed as external dependencies:
  * [SEAL](https://www.microsoft.com/en-us/research/project/simple-encrypted-arithmetic-library/)
  * [HEAAN](https://eprint.iacr.org/2016/421.pdf)

## Building HE Transformer with nGraph

```
# Clone
git clone https://github.com/NervanaSystems/he-transformer.git
cd he-transformer
mkdir build
cd build

# Config
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Build
make -j$(nproc)

# HEAAN unit tests: `HE_HEAAN.*`
./test/unit-test --gtest_filter="HE_HEAAN.ab"

# SEAL unit tests: `HE_SEAL.*`
./test/unit-test --gtest_filter="HE_SEAL.ab"
```

## How to manage `he-transformer` and `ngraph` repos

- Clone `he-transformer` and build
- `ngraph` repo will be located at `he-transformer/build/ext_ngraph/src/ext_ngraph`
- In `ngraph` repo, we can make modifications, and perform git operations
- In `he-transformer` repo, the build scripts will pick up the changes inside `ngraph` repo

## Code formatting

Please run `maint/apply-code-format.sh` before submitting a PR.

## Examples
The `examples` directory contains two deep learning examples which depend on [nGraph-Tensorflow-bridge](https://github.com/NervanaSystems/ngraph-tensorflow-bridge/).

 - `mnist_softmax_ngraph.py` is a simple one-hidden layer feedforward neural network, and should be run as `python mnist_softmax_ngraph.py`
  - `mnist_deep_simplified_he.py` is a more complicated convolutional neural network (CNN) which achieves ~99% accuracy. The model is based on [Cryptonets](http://proceedings.mlr.press/v48/gilad-bachrach16.pdf).
       - This should be run as `XLA_NGRAPH_ENABLE_SERIALIZE=1 python mnist_deep_simplified_he.py` and will export several `.js` files which are serialized nGraph computation graphs as well as save the model weights in `.txt` and `.bin` files. To use pre-computed weights, skip this step.
       - Copy the correct `.js` file to `test/model/mnist_cryptonets_batch_$(batch_size).js` where `$(batch_size)$` should be a power of two in 2, 4, 1024, 4096. To identify the correct `.js` file, note the shape of the parameters; one of the parameters in the correct file should have  `shape : [batch_size, 784]`. Also copy the weights to `test/model/weights/`
       - Run `make -j unit-test` from the `build` directory
       - Run `./test/unit-test --gtest_filter="HE_HEAAN.tf_mnist_cryptonets_batch"` to run the CryptoNets example on MNIST using the HEAAN backend. Note, for optimal performance, install OpenMP and set `OMP_NUM_THREADS=$(nproc)`.
