# HE Transformer for nGraph

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

# nGraph backend unit tests: `HE.*`
./test/unit-test --gtest_filter="HE.ab"

# HE specific unit tests: `TestHEBackend.*`
./test/unit-test --gtest_filter="TestHEBackend.ab"
```

## How to manage `he-transformer` and `ngraph` repos

- Clone `he-transformer` and build
- `ngraph` repo will be located at `he-transformer/build/ext_ngraph/src/ext_ngraph`
- In `ngraph` repo, we can make modifications, and perform git operations
- In `he-transformer` repo, the build scripts will pick up the changes inside `ngraph` repo

## Code formatting

Please run `maint/apply-code-format.sh` before submitting a PR.
