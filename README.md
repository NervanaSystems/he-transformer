# HE Transformer for nGraph

The **Intel® HE transformer for nGraph™** is a Homomorphic Encryption (HE) backend to the [**Intel® nGraph Compiler**](https://github.com/NervanaSystems/ngraph), Intel's graph compiler for Artificial Neural Networks.

Homomorphic encryption is a form of encryption that allows computation on encrypted data, and is an attractive remedy to increasing concerns about data privacy in the field of machine learning. For more information, see our [paper](https://arxiv.org/pdf/1810.10121.pdf).

This project is meant as a proof-of-concept to demonstrate the feasibility of HE  on local machines. The goal is to measure performance of various HE schemes for deep learning. This is  **not** intended to be a production-ready product, but rather a research tool.

Currently, we support two encryption schemes, both provided by the [SEAL encryption library](https://www.microsoft.com/en-us/research/project/simple-encrypted-arithmetic-library/), which is installed as an external dependency:
  * [BFV](https://eprint.iacr.org/2016/510.pdf)
  * [CKKS](https://eprint.iacr.org/2018/931.pdf)

Additionally, we integrate with the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/NervanaSystems/ngraph-tf) to allow users to run inference on trained neural networks through Tensorflow.

## Building HE Transformer

See the [wiki](https://github.com/NervanaSystems/he-transformer/wiki) for information on how to build the he-transformer.

## Code formatting

Please run `maint/apply-code-format.sh` before submitting a pull request.

## Examples
The `examples` directory contains a deep learning example which depends on the [**Intel® nGraph™ Compiler and runtime engine for TensorFlow**](https://github.com/NervanaSystems/ngraph-tf).
