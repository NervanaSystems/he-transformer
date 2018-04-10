# HE Transformer for Ngraph

## Building Ngraph with HE Transformer

```
git clone git@github.com:NervanaSystems/ngraph-he.git
cd ngraph-he
git checkout he-master # the `master` branch is the upstream ngraph, `he-master` is our master
mkdir build
cd build

cmake -DNGRAPH_HE_ENABLE=True ..
make -j$(nproc)
```

## Managing `ngraph-he` and `he-transformer` repos

- Only need to clone `ngraph-he` manually
- `he-transformer` repo will be located at
  `ngraph-he/build/third-party/he_transformer/src/ext_he_transformer`
- In `he-transformer` repo, we can make changes and do git stuffs independent from `ngraph-he`
- `ngraph-he`'s build system will pick up changes in the `he-transformer` repo
