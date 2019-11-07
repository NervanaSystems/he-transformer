


1) Ubuntu 16.04 with clang-6


```bash
export $HE_TRANSFORMER=$(pwd)
mkdir build
cd build
cmake .. -DCMAKE_CXX_COMPILER=clang++-6.0 -DCMAKE_C_COMPILER=clang-6.0
make install

# No failures
./test/unit-test

source external/venv-tf-py3/bin/activate
make python_client
pip install python/dist/pyhe_client-*.whl

# Don't validate output; just check it runs
python -c "import pyhe_client"
python $HE_TRANSFORMER/examples/ax.py --backend=CPU
python $HE_TRANSFORMER/examples/ax.py --backend=HE_SEAL
```

2) Ubuntu 18.04 with gcc7

```bash
export $HE_TRANSFORMER=$(pwd)
mkdir build
cd build
cmake .. -DCMAKE_CXX_COMPILER=g++-7 -DCMAKE_C_COMPILER=gcc-7
make install

# No failures
./test/unit-test

source external/venv-tf-py3/bin/activate
make python_client
pip install python/dist/pyhe_client-*.whl

# Don't validate output; just check it runs
python -c "import pyhe_client"
python $HE_TRANSFORMER/examples/ax.py --backend=CPU
python $HE_TRANSFORMER/examples/ax.py --backend=HE_SEAL
```

3) Ubuntu 18.04 with clang-9; build docs and check formatting
```bash
export $HE_TRANSFORMER=$(pwd)
mkdir build
cd build
cmake .. -DCMAKE_CXX_COMPILER=clang-9 -DNGRAPH_HE_DOC_BUILD_ENABLE=ON -DNGRAPH_HE_CLANG_TIDY=ON

# No incorrectly-formatted files
make style-check

# No failures
make docs

# No clang-tidy errors
make install

# No failures
./test/unit-test

source external/venv-tf-py3/bin/activate
make python_client
pip install python/dist/pyhe_client-*.whl

# Don't validate output; just check it runs
python -c "import pyhe_client"
python $HE_TRANSFORMER/examples/ax.py --backend=CPU
python $HE_TRANSFORMER/examples/ax.py --backend=HE_SEAL
```