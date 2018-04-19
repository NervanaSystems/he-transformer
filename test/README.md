### HE Transformer Test through Ngraph

- The tests in this directory are bundled in Ngraph's `unit-test` executable.
- They are used to test HE Transformer through Ngraph's public API.
- The `unit-test` executable only links with `libngraph.so`. It does not link HE libraries (such
  as SEAL) directly. The only way to use SEAL is by using ngraph's public API through HE
  transformer.

To run `unit-test` for HE transformer:

```
# In `ngraph-he/build` directory
./test/unit-test --gtest_filter="HE.*" # (The gtest filter is for illustration purpose only.s)
```
