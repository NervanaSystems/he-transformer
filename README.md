<p align="center">
  <img src="images/nGraph_mask_1-1.png" width="200">
</p>

# HE Transformer for nGraph

The **Intel® HE transformer for nGraph™** is a Homomorphic Encryption (HE) backend to the [**Intel® nGraph Compiler**](https://github.com/NervanaSystems/ngraph), Intel's graph compiler for Artificial Neural Networks.

Homomorphic encryption is a form of encryption that allows computation on encrypted data, and is an attractive remedy to increasing concerns about data privacy in the field of machine learning. For more information, see our [original paper](https://arxiv.org/pdf/1810.10121.pdf). Our [updated paper](https://arxiv.org/pdf/1908.04172.pdf) showcases many of the recent advances in he-transformer.

This project is meant as a proof-of-concept to demonstrate the feasibility of HE  on local machines. The goal is to measure performance of various HE schemes for deep learning. This is  **not** intended to be a production-ready product, but rather a research tool.

Currently, we support the [CKKS](https://eprint.iacr.org/2018/931.pdf) encryption scheme, implemented by the [Simple Encrypted Arithmetic Library (SEAL)](https://github.com/Microsoft/SEAL) from Microsoft Research.

Additionally, we integrate with the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/tensorflow/ngraph-bridge) to allow users to run inference on trained neural networks through Tensorflow.

## Examples
The [examples](https://github.com/NervanaSystems/he-transformer/tree/master/examples) folder contains a deep learning example which depends on the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/tensorflow/ngraph-bridge).

## Building HE Transformer

### Dependencies
- Operating system: Ubuntu 16.04, Ubuntu 18.04.
- CMake >= 3.12
- Compiler: g++ version >= 6.0, clang >= 5.0
- OpenMP is strongly suggested, though not strictly necessary. You may experience slow runtimes without OpenMP
- python3 and pip3
- virtualenv v16.1.0
- bazel v0.25.2
#### The following dependencies are built automatically
- [nGraph](https://github.com/NervanaSystems/ngraph) - v0.25.1-rc.8
- [nGraph-tf](https://github.com/tensorflow/ngraph-bridge) - commit cad093d84cc3a1ce212d8a96c67217321b44309b
- [SEAL](https://github.com/Microsoft/SEAL) - v3.4.2
- [TensorFlow](https://github.com/tensorflow/tensorflow) - v1.14.0
- [Boost](https://github.com/boostorg) v1.69
- [Google protobuf](https://github.com/protocolbuffers/protobuf) v3.9.1

We also offer [docker containers](https://github.com/NervanaSystems/he-transformer/tree/master/contrib/docker) and builds of he-transformer on a reference OS.

### To install bazel
```bash
    wget https://github.com/bazelbuild/bazel/releases/download/0.25.2/bazel-0.25.2-installer-linux-x86_64.sh
    bash bazel-0.25.2-installer-linux-x86_64.sh --user
 ```
 Add and source the bin path to your `~/.bashrc` file to call bazel
```bash
 export PATH=$PATH:~/bin
 source ~/.bashrc
```

### 1. Build HE-Transformer
Before building, make sure you deactivate any active virtual environments (i.e. run `deactivate`)
```bash
git clone https://github.com/NervanaSystems/he-transformer.git
cd he-transformer
export HE_TRANSFORMER=$(pwd)
mkdir build
cd $HE_TRANSFORMER/build
cmake .. -DCMAKE_CXX_COMPILER=g++-7 -DCMAKE_C_COMPILER=gcc-7
make install
source external/venv-tf-py3/bin/activate
```

#### 1a. To build documentation
First install doxygen, i.e.
```bash
sudo apt-get install doxygen
```
Then add the following CMake flag
```bash
cmake .. -DNGRAPH_HE_DOC_BUILD_ENABLE=ON
```
and call
```bash
make docs
```
to create doxygen documentation in `$HE_TRANSFORMER/build/doc/doxygen`.

#### 1b. Python bindings for client
To build an client-server model with python bindings, see the [python](https://github.com/NervanaSystems/he-transformer/tree/master/python) folder.

### 2. Run C++ unit-tests
Ensure the virtual environment is active, i.e. run `source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate`
```bash
cd $HE_TRANSFORMER/build
# To run single HE_SEAL unit-test
./test/unit-test --gtest_filter="HE_SEAL.add_2_3"
# To run all C++ unit-tests
./test/unit-test
```

### 3. Run Simple python example
Ensure the virtual environment is active, i.e. run `source $HE_TRANSFORMER/build/external/venv-tf-py3/bin/activate`
```bash
cd $HE_TRANSFORMER/examples
# Run with CPU
python ax.py
# To run CKKS unit-test
NGRAPH_TF_BACKEND=HE_SEAL python ax.py
```

For a deep learning example, see [examples/MNIST/Cryptonets/](https://github.com/NervanaSystems/he-transformer/tree/master/examples/MNIST/Cryptonets).

## Code formatting
Please run `maint/apply-code-format.sh` before submitting a pull request.