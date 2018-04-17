### HE Transformer Test and HE Library Test

- The tests in this directory are bundled in the `test-he` executable.
- They are used and only used to test HE transformer without going through Ngraph's public API.
- The `unit-test` executable links
  - `libngraph.so`
  - `libhe.so`
  - Other libraries used by HE transformer

To run `test-he` for HE transformer:

```
# In `ngraph-he/build` directory
./test/test-he
```
