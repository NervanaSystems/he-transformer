# HE Transformer for nGraph

The **Intel® HE transformer for nGraph™** is a Homomorphic Encryption (HE) backend to the [**Intel® nGraph Compiler**](https://github.com/NervanaSystems/ngraph), Intel's graph compiler for Artificial Neural Networks.

Homomorphic encryption is a form of encryption that allows computation on encrypted data, and is an attractive remedy to increasing concerns about data privacy in the field of machine learning. For more information, see our [paper](https://arxiv.org/pdf/1810.10121.pdf).

This project is meant as a proof-of-concept to demonstrate the feasibility of HE  on local machines. The goal is to measure performance of various HE schemes for deep learning. This is  **not** intended to be a production-ready product, but rather a research tool.

Currently, we support two encryption schemes, both provided by the [SEAL encryption library](https://www.microsoft.com/en-us/research/project/simple-encrypted-arithmetic-library/), which is installed as an external dependency:
  * [BFV](https://eprint.iacr.org/2016/510.pdf)
  * [CKKS](https://eprint.iacr.org/2018/931.pdf)

Additionally, we integrate with the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/NervanaSystems/ngraph-tf) to allow users to run inference on trained neural networks through Tensorflow.

## Examples
The [examples](https://github.com/NervanaSystems/he-transformer/tree/master/examples) directory contains a deep learning example which depends on the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/NervanaSystems/ngraph-tf).


## Building HE Transformer

### Dependencies
- We currently only support Ubuntu 16.04
- CMake 3.12.3, although different versions may work
- Clang version 6.0, although different versions may work
- OpenMP is strongly suggested, though not strictly necessary. You may experience slow runtimes without OpenMP
#### The following dependencies are built automatically
- [nGraph](https://github.com/NervanaSystems/ngraph) version 0.10.1
- [nGraph-tf](https://github.com/NervanaSystems/ngraph-tf) - For Tensorflow integration

The `docker` folder contains a script to build `he-transformer` through Docker for testing purposes, and requires only Docker.

### 1. If the setup has already been done
TLDR, if the setup has already been done, here's the command to run the python example and C++ unit-tests.
```bash
# Replace ~/repos/he-transformer with wherever you installed he-transformer
export HE_TRANSFORMER=~/repos/he-transformer
```
#### Without Tensorflow (TF)
```bash
cd $HE_TRANSFORMER/build
./test/unit-test --gtest_filter="HE_SEAL_BFV.add_2_3"
./test/unit-test --gtest_filter="HE_SEAL_CKKS.add_2_3"
```

#### With TF
```bash
# activate python environment with Tensorflow and nGraph installed
source ~/repos/venvs/he3/bin/activate
cd $HE_TRANSFORMER/examples/
# run on SEAL BFV backend
NGRAPH_TF_BACKEND=HE:SEAL:BFV python axpy.py
# run on SEAL CKKS backend
NGRAPH_TF_BACKEND=HE:SEAL:CKKS python axpy.py
```

### 2. Build `he-transformer`
- Clone the repository
```bash
git clone https://github.com/NervanaSystems/he-transformer.git
cd he-transformer
export HE_TRANSFORMER=$(pwd)
```
- Create build directory
```bash
mkdir $HE_TRANSFORMER/build
cd $HE_TRANSFORMER/build
```
#### 2.1a Build HE Transformer without TF
-  To build without TF, run
```bash
cmake .. [-DCMAKE_CXX_COMPILER=clang++-6.0 -DCMAKE_C_COMPILER=clang-6.0]
make -j
```
#### 2.1b Build HE Transformer with TF
 To build with TF, first create and activate python virtual environment
```bash
mkdir -p ~/venvs
virtualenv ~/venvs/he3 -p python3
source ~/venvs/he3/bin/activate
```
and then, with the python environment active, run
```bash
cmake .. -DENABLE_TF=ON [-DCMAKE_CXX_COMPILER=clang++-6.0 -DCMAKE_C_COMPILER=clang-6.0]
make -j
make -j install
```

#### 2.2 Run C++ Native example
Whether or not you built with TF support (2.1a or 2.1b), you can run the C++ unit-tests.
```bash
cd $HE_TRANSFORMER/build
./test/unit-test --gtest_filter="HE_SEAL_BFV.add_2_3"
./test/unit-test --gtest_filter="HE_SEAL_CKKS.add_2_3"
```
This will run a basic addition unit-test with the SEAL BFV and SEAL CKKS backends. To run all the C++ unit tests,
```bash
./test/unit-test
```

#### 2.3 Run TF python model

Note: must be done inside the python virtual environment from step 2.1b.

```bash
# source activate virtualenv first
source ~/venvs/he3/bin/activate
cd $HE_TRANSFORMER/examples

# run on ngraph-CPU
python axpy.py

# run on SEAL:BFV
NGRAPH_TF_BACKEND=HE:SEAL:BFV python axpy.py

# run on SEAL:CKKS
NGRAPH_TF_BACKEND=HE:SEAL:CKKS python axpy.py
```

For a deep learning example, see [examples/cryptonets/](https://github.com/NervanaSystems/he-transformer/tree/master/examples/cryptonets).

## Code formatting

Please run `maint/apply-code-format.sh` before submitting a pull request.

