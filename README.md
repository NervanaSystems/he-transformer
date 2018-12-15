# HE Transformer for nGraph

The **Intel® HE transformer for nGraph™** is a Homomorphic Encryption (HE) backend to the [**Intel® nGraph Compiler**](https://github.com/NervanaSystems/ngraph), Intel's graph compiler for Artificial Neural Networks.

Homomorphic encryption is a form of encryption that allows computation on encrypted data, and is an attractive remedy to increasing concerns about data privacy in the field of machine learning. For more information, see our [paper](https://arxiv.org/pdf/1810.10121.pdf).

This project is meant as a proof-of-concept to demonstrate the feasibility of HE  on local machines. The goal is to measure performance of various HE schemes for deep learning. This is  **not** intended to be a production-ready product, but rather a research tool.

Currently, we support two encryption schemes, implemented by the [Simple Encrypted Arithmetic Library (SEAL)](https://github.com/Microsoft/SEAL) from Microsoft Research:
  * [BFV](https://eprint.iacr.org/2016/510.pdf)
  * [CKKS](https://eprint.iacr.org/2018/931.pdf)

Additionally, we integrate with the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/NervanaSystems/ngraph-tf) to allow users to run inference on trained neural networks through Tensorflow.

## Examples
The [examples](https://github.com/NervanaSystems/he-transformer/tree/master/examples) directory contains a deep learning example which depends on the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/NervanaSystems/ngraph-tf).

## Building HE Transformer

### Dependencies
- We currently only support Ubuntu 16.04
- CMake >= 3.10, although different versions may work
- GCC version 7, although different versions may work
- OpenMP is strongly suggested, though not strictly necessary. You may experience slow runtimes without OpenMP
- virtualenv
- bazel v0.16.0
#### The following dependencies are built automatically
- [nGraph](https://github.com/NervanaSystems/ngraph) - v0.11.0
- [nGraph-tf](https://github.com/NervanaSystems/ngraph-tf) - v0.9.0
- [SEAL](https://github.com/Microsoft/SEAL) version 3.1

### To install bazel
```bash
 wget https://github.com/bazelbuild/bazel/releases/download/0.16.0/bazel-0.16.0-installer-linux-x86_64.sh
 chmod +x bazel-0.16.0-installer-linux-x86_64.sh
 ./bazel-0.16.0-installer-linux-x86_64.sh --user
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
cmake .. [-DCMAKE_CXX_COMPILER=g++-7 -DCMAKE_C_COMPILER=gcc-7]
make -j install
source external/venv-tf-py3/bin/activate
```

### 2. Run C++ unit-tests
Ensure the virtual environment is active, i.e. run `source $HE_TRANSFORMER /external/venv-tf-py3/bin/activate`
```bash
cd $HE_TRANSFORMER/build
# To run CKKS unit-test
./test/unit-test --gtest_filter="HE_SEAL_CKKS.*abc*"
# To run BFV unit-test
./test/unit-test --gtest_filter="HE_SEAL_BFV.*abc*
# To run all C++ unit-tests
./test/unit-test
```

### 3. Run Simple python example
Ensure the virtual environment is active, i.e. run `source $HE_TRANSFORMER /external/venv-tf-py3/bin/activate`
```bash
cd $HE_TRANSFORMER/examples
# Run with CPU
python axpy.py
# To run CKKS unit-test
NGRAPH_TF_BACKEND=HE_SEAL_CKKS python axpy.py
# To run BFV unit-test
NGRAPH_TF_BACKEND=HE_SEAL_BFV python axpy.py
```

For a deep learning example, see [examples/cryptonets/](https://github.com/NervanaSystems/he-transformer/tree/master/examples/cryptonets).

## Code formatting
Please run `maint/apply-code-format.sh` before submitting a pull request.
