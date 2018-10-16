# HE Transformer for nGraph

HE transformer is a Homomorphic Encryption (HE) backend to nGraph.
This is meant as a Proof-of-concept project to demonstrate the feasibility of homomorphic encryption (HE) on **local** machines.

The goal is to measure performance of various HE schemes for deep learning.

This is  **not** intended to be a production-ready product, but rather a research tool.

Currently, we support two backends, which are installed as external dependencies:
  * [SEAL](https://www.microsoft.com/en-us/research/project/simple-encrypted-arithmetic-library/)
  * [HEAAN](https://eprint.iacr.org/2016/421.pdf)

## Building HE Transformer with nGraph

See the [wiki](https://github.com/NervanaSystems/he-transformer/wiki) for information on how to build ngraph-he.

## Code formatting

Please run `maint/apply-code-format.sh` before submitting a PR.

## Examples
The `examples` directory contains a deep learning example which depends on [nGraph-Tensorflow](https://github.com/NervanaSystems/ngraph-tf).
