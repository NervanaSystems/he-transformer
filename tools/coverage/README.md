# Code coverage

See (here)[https://alastairs-place.net/blog/2016/05/20/code-coverage-from-the-command-line-with-clang/] for a brief introduction to the different profiling options.

```bash
# Enable coverage compile options
cmake .. -DCMAKE_CXX_COMPILER=clang++-9 -DCMAKE_INSTALL_PREFIX=~/mylibs -DCMAKE_C_COMPILER=clang-9 -DNGRAPH_HE_CODE_COVERAGE=ON -DNGRAPH_HE_CLANG_TIDY=OFF


# Create raw coverage file
# Use verbose logging to increase coverage
LLVM_PROFILE_FILE="unit_test.profraw" NGRAPH_HE_LOG_LEVEL=5 NGRAPH_HE_VERBOSE_OPS=all ./test/unit-test

# Convert to format llvm-cov wants
llvm-profdata-9 merge -o testcov.profdata unit_test.profraw

# Generate coverage report for just he-transformer src files
llvm-cov-9 report src/libhe_seal_backend.so --use-color=1 --instr-profile=testcov.profdata ../src/* > coverage.report

# Generate html coverage report
llvm-cov-9 show src/libhe_seal_backend.so --instr-profile=testcov.profdata ../src/*  -Xdemangler=c++filt --format=html > cov.html