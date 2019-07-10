# HE Transformer for nGraph

The **Intel® HE transformer for nGraph™** is a Homomorphic Encryption (HE) backend to the [**Intel® nGraph Compiler**](https://github.com/NervanaSystems/ngraph), Intel's graph compiler for Artificial Neural Networks.

Homomorphic encryption is a form of encryption that allows computation on encrypted data, and is an attractive remedy to increasing concerns about data privacy in the field of machine learning. For more information, see our [paper](https://arxiv.org/pdf/1810.10121.pdf).

This project is meant as a proof-of-concept to demonstrate the feasibility of HE  on local machines. The goal is to measure performance of various HE schemes for deep learning. This is  **not** intended to be a production-ready product, but rather a research tool.

Currently, we support the [CKKS](https://eprint.iacr.org/2018/931.pdf) encryption scheme, implemented by the [Simple Encrypted Arithmetic Library (SEAL)](https://github.com/Microsoft/SEAL) from Microsoft Research.

Additionally, we integrate with the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/tensorflow/ngraph-bridge) to allow users to run inference on trained neural networks through Tensorflow.

## Examples
The [examples](https://github.com/NervanaSystems/he-transformer/tree/master/examples) directory contains a deep learning example which depends on the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/tensorflow/ngraph-bridge).

## Building HE Transformer

### Dependencies
- We currently only support Ubuntu 16.04. See below for possible ways to compile on Ubuntu 18.04
- CMake >= 3.10, although different versions may work
- GCC version 7, although different versions may work
- OpenMP is strongly suggested, though not strictly necessary. You may experience slow runtimes without OpenMP
- python3.5 and pip3
- virtualenv v16.1.0
- bazel v0.21.0
#### The following dependencies are built automatically
- [nGraph](https://github.com/NervanaSystems/ngraph) - v0.19.1
- [nGraph-tf](https://github.com/tensorflow/ngraph-bridge) - v0.14.0
- [SEAL](https://github.com/Microsoft/SEAL) - v3.3
- [TensorFlow](https://github.com/tensorflow/tensorflow) - v1.13.1
- Boost 1.69

### To install bazel
```bash
 wget https://github.com/bazelbuild/bazel/releases/download/0.21.0/bazel-0.21.0-installer-linux-x86_64.sh
 chmod +x bazel-0.21.0-installer-linux-x86_64.sh
 ./bazel-0.21.0-installer-linux-x86_64.sh --user
 ```
 Make sure to add and source the bin path to your `~/.bashrc` file in order to be able to call bazel from the user's installation we set up:
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

For Ubuntu 18.04, you may try adding `-DPYTHON_VENV_VERSION=python3.6` to the cmake command above.

The first build will compile Tensorflow. To speed up subsequent builds, you can avoid compiling Tensorflow by calling
```bash
cmake .. -DUSE_PREBUILT_TF=ON -DCMAKE_CXX_COMPILER=g++-7 -DCMAKE_C_COMPILER=gcc-7
```

#### 1.b Python bindings for client
To build an experimental client-server model with python bindings, see the `python` folder.
***Note***: This feature is experimental. For best experience, you should skip this step.

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

For a deep learning example, see [examples/cryptonets/](https://github.com/NervanaSystems/he-transformer/tree/master/examples/MNIST-Cryptonets).

## Code formatting
Please run `maint/apply-code-format.sh` before submitting a pull request.
