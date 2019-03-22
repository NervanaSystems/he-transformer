This folder contains the files necessary to reproduce benchmarks results found in publications.

* CryptoNets - contains runtimes for MNIST-Cryptonets network (Table 3, Table 4, Figure 6).
* GEMM - contains results for sparse matrix multiplication (Figure 5)
* CIFAR10 - contains results for CIFAR-10 models (Table 6)

For binarized CryptoNets results (Table 5), see the `examples/MNIST-BNN-BN` folder

Performance analysis completed on Jan 16, 2019 - Mar 21, 2019 by Intel using a Xeon Platinum 8180 platform with 112 CPUs operating at 2.5Ghz, 2 sockets, and 376GB of RAM running HE Transformer (v0.2, with the modifications in the `benchmarks` branch) with nGraph-tf (v 0.9.0) and nGraph (v 0.11.0) on Ubuntu 16.04.4 LTS