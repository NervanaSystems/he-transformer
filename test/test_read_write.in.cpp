//*****************************************************************************
// Copyright 2018-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <climits>

#include "ngraph/ngraph.hpp"
#include "seal/he_seal_backend.hpp"
#include "test_util.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;
using namespace ngraph::he;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, plain_tv_write_read_scalar) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());

  Shape shape{};
  {
    auto a = he_backend->create_plain_tensor(element::f32, shape);
    copy_data(a, vector<float>{5});
    EXPECT_TRUE(test::he::all_close(read_vector<float>(a), (vector<float>{5})));
  }
  {
    auto a = he_backend->create_plain_tensor(element::f64, shape);
    copy_data(a, vector<double>{5});
    EXPECT_TRUE(
        test::he::all_close(read_vector<double>(a), (vector<double>{5})));
  }
  {
    auto a = he_backend->create_plain_tensor(element::i64, shape);
    copy_data(a, vector<int64_t>{5});
    EXPECT_TRUE(
        test::he::all_close(read_vector<int64_t>(a), (vector<int64_t>{5})));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, plain_tv_write_read_large_scalar_int64) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());

  Shape shape{};
  auto a = he_backend->create_plain_tensor(element::i64, shape);
  copy_data(a, vector<int64_t>{LLONG_MAX});
  EXPECT_TRUE(test::he::all_close(read_vector<int64_t>(a),
                                  (vector<int64_t>{LLONG_MAX})));

  copy_data(a, vector<int64_t>{LLONG_MIN});
  EXPECT_TRUE(test::he::all_close(read_vector<int64_t>(a),
                                  (vector<int64_t>{LLONG_MIN})));
}

NGRAPH_TEST(${BACKEND_NAME}, plain_tv_write_read_2) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());

  Shape shape{2};
  {
    auto a = he_backend->create_plain_tensor(element::f32, shape);
    copy_data(a, vector<float>{5, 6});
    EXPECT_TRUE(
        test::he::all_close(read_vector<float>(a), (vector<float>{5, 6})));
  }
  {
    auto a = he_backend->create_plain_tensor(element::f64, shape);
    copy_data(a, vector<double>{5, 6});
    EXPECT_TRUE(
        test::he::all_close(read_vector<double>(a), (vector<double>{5, 6})));
  }
  {
    auto a = he_backend->create_plain_tensor(element::i64, shape);
    copy_data(a, vector<int64_t>{5, 6});
    EXPECT_TRUE(
        test::he::all_close(read_vector<int64_t>(a), (vector<int64_t>{5, 6})));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, plain_tv_write_read_2_3) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());

  Shape shape{2, 3};
  {
    auto a = he_backend->create_plain_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 2>({{1, 2}, {3, 4}, {5, 6}}).get_vector());
    EXPECT_TRUE(test::he::all_close(
        read_vector<float>(a),
        test::NDArray<float, 2>({{1, 2}, {3, 4}, {5, 6}}).get_vector()));
  }
  {
    auto a = he_backend->create_plain_tensor(element::f64, shape);
    copy_data(a,
              test::NDArray<double, 2>({{1, 2}, {3, 4}, {5, 6}}).get_vector());
    EXPECT_TRUE(test::he::all_close(
        read_vector<double>(a),
        test::NDArray<double, 2>({{1, 2}, {3, 4}, {5, 6}}).get_vector()));
  }
  {
    auto a = he_backend->create_plain_tensor(element::i64, shape);
    copy_data(a,
              test::NDArray<int64_t, 2>({{1, 2}, {3, 4}, {5, 6}}).get_vector());
    EXPECT_TRUE(test::he::all_close(
        read_vector<int64_t>(a),
        test::NDArray<int64_t, 2>({{1, 2}, {3, 4}, {5, 6}}).get_vector()));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, plain_tv_batch_write_read_2_3) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());

  Shape shape{2, 3};
  {
    auto a = he_backend->create_packed_plain_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 2>({{1, 2}, {3, 4}, {5, 6}}).get_vector());
    EXPECT_TRUE(test::he::all_close(
        read_vector<float>(a),
        (test::NDArray<float, 2>({{1, 2}, {3, 4}, {5, 6}})).get_vector()));
  }
  {
    auto a = he_backend->create_packed_plain_tensor(element::f64, shape);
    copy_data(a,
              test::NDArray<double, 2>({{1, 2}, {3, 4}, {5, 6}}).get_vector());
    EXPECT_TRUE(test::he::all_close(
        read_vector<double>(a),
        (test::NDArray<double, 2>({{1, 2}, {3, 4}, {5, 6}})).get_vector()));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, cipher_tv_write_read_scalar) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());

  Shape shape{};
  {
    auto a = he_backend->create_cipher_tensor(element::f32, shape);
    copy_data(a, vector<float>{5});
    EXPECT_TRUE(test::he::all_close(read_vector<float>(a), (vector<float>{5})));
  }
  {
    auto a = he_backend->create_cipher_tensor(element::f64, shape);
    copy_data(a, vector<double>{5});
    EXPECT_TRUE(
        test::he::all_close(read_vector<double>(a), (vector<double>{5})));
  }
  {
    NGRAPH_INFO << "int 32";
    auto a = he_backend->create_cipher_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{5});
    EXPECT_TRUE(
        test::he::all_close(read_vector<int32_t>(a), (vector<int32_t>{5})));
  }
  {
    NGRAPH_INFO << "int 64";
    auto a = he_backend->create_cipher_tensor(element::i64, shape);
    copy_data(a, vector<int64_t>{5});
    EXPECT_TRUE(
        test::he::all_close(read_vector<int64_t>(a), (vector<int64_t>{5})));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, cipher_tv_write_read_2) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());
  Shape shape{2};
  {
    auto a = he_backend->create_cipher_tensor(element::f32, shape);
    copy_data(a, vector<float>{5, 6});
    EXPECT_TRUE(
        test::he::all_close(read_vector<float>(a), (vector<float>{5, 6})));
  }
  {
    auto a = he_backend->create_cipher_tensor(element::f64, shape);
    copy_data(a, vector<double>{5, 6});
    EXPECT_TRUE(
        test::he::all_close(read_vector<double>(a), (vector<double>{5, 6})));
  }
  {
    auto a = he_backend->create_cipher_tensor(element::i64, shape);
    copy_data(a, vector<int64_t>{5, 6});
    EXPECT_TRUE(
        test::he::all_close(read_vector<int64_t>(a), (vector<int64_t>{5, 6})));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, cipher_tv_write_read_2_complex) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());
  he_backend->update_encryption_parameters(
      HESealEncryptionParameters::default_complex_packing_parms());

  Shape shape{2};
  {
    auto a = he_backend->create_cipher_tensor(element::f32, shape);
    copy_data(a, vector<float>{5, 6});
    EXPECT_TRUE(
        test::he::all_close(read_vector<float>(a), (vector<float>{5, 6})));
  }
  {
    auto a = he_backend->create_cipher_tensor(element::f64, shape);
    copy_data(a, vector<double>{5, 6});
    EXPECT_TRUE(
        test::he::all_close(read_vector<double>(a), (vector<double>{5, 6})));
  }
  {
    auto a = he_backend->create_cipher_tensor(element::i64, shape);
    copy_data(a, vector<int64_t>{5, 6});
    EXPECT_TRUE(
        test::he::all_close(read_vector<int64_t>(a), (vector<int64_t>{5, 6})));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, cipher_tv_write_read_2_3) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());

  Shape shape{2, 3};
  {
    auto a = he_backend->create_cipher_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 2>({{1, 2}, {3, 4}, {5, 6}}).get_vector());
    EXPECT_TRUE(test::he::all_close(
        read_vector<float>(a),
        (test::NDArray<float, 2>({{1, 2}, {3, 4}, {5, 6}})).get_vector()));
  }
  {
    auto a = he_backend->create_cipher_tensor(element::f64, shape);
    copy_data(a,
              test::NDArray<double, 2>({{1, 2}, {3, 4}, {5, 6}}).get_vector());
    EXPECT_TRUE(test::he::all_close(
        read_vector<double>(a),
        (test::NDArray<double, 2>({{1, 2}, {3, 4}, {5, 6}})).get_vector()));
  }
  {
    auto a = he_backend->create_cipher_tensor(element::i64, shape);
    copy_data(a,
              test::NDArray<int64_t, 2>({{1, 2}, {3, 4}, {5, 6}}).get_vector());
    EXPECT_TRUE(test::he::all_close(
        read_vector<int64_t>(a),
        (test::NDArray<int64_t, 2>({{1, 2}, {3, 4}, {5, 6}})).get_vector()));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, cipher_tv_write_read_5_5) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());

  Shape shape{5, 5};
  {
    auto a = he_backend->create_cipher_tensor(element::f32, shape);
    copy_data(a, test::NDArray<float, 2>({{1, 2, 3, 4, 5},
                                          {6, 7, 8, 9, 10},
                                          {11, 12, 13, 14, 15},
                                          {16, 17, 18, 19, 20},
                                          {21, 22, 23, 24, 25}})
                     .get_vector());
    EXPECT_TRUE(test::he::all_close(
        read_vector<float>(a), (test::NDArray<float, 2>({{1, 2, 3, 4, 5},
                                                         {6, 7, 8, 9, 10},
                                                         {11, 12, 13, 14, 15},
                                                         {16, 17, 18, 19, 20},
                                                         {21, 22, 23, 24, 25}}))
                                   .get_vector()));
  }
  {
    auto a = he_backend->create_cipher_tensor(element::f64, shape);
    copy_data(a, test::NDArray<double, 2>({{1, 2, 3, 4, 5},
                                           {6, 7, 8, 9, 10},
                                           {11, 12, 13, 14, 15},
                                           {16, 17, 18, 19, 20},
                                           {21, 22, 23, 24, 25}})
                     .get_vector());
    EXPECT_TRUE(
        test::he::all_close(read_vector<double>(a),
                            (test::NDArray<double, 2>({{1, 2, 3, 4, 5},
                                                       {6, 7, 8, 9, 10},
                                                       {11, 12, 13, 14, 15},
                                                       {16, 17, 18, 19, 20},
                                                       {21, 22, 23, 24, 25}}))
                                .get_vector()));
  }
  {
    auto a = he_backend->create_cipher_tensor(element::i64, shape);
    copy_data(a, test::NDArray<int64_t, 2>({{1, 2, 3, 4, 5},
                                            {6, 7, 8, 9, 10},
                                            {11, 12, 13, 14, 15},
                                            {16, 17, 18, 19, 20},
                                            {21, 22, 23, 24, 25}})
                     .get_vector());
    EXPECT_TRUE(
        test::he::all_close(read_vector<int64_t>(a),
                            (test::NDArray<int64_t, 2>({{1, 2, 3, 4, 5},
                                                        {6, 7, 8, 9, 10},
                                                        {11, 12, 13, 14, 15},
                                                        {16, 17, 18, 19, 20},
                                                        {21, 22, 23, 24, 25}}))
                                .get_vector()));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, cipher_tv_batch_write_read_2_3) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());

  Shape shape{2, 3};
  {
    auto a = he_backend->create_packed_cipher_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 2>({{1, 2}, {3, 4}, {5, 6}}).get_vector());
    EXPECT_TRUE(test::he::all_close(
        read_vector<float>(a),
        (test::NDArray<float, 2>({{1, 2}, {3, 4}, {5, 6}})).get_vector()));
  }
  {
    auto a = he_backend->create_packed_cipher_tensor(element::f64, shape);
    copy_data(a,
              test::NDArray<double, 2>({{1, 2}, {3, 4}, {5, 6}}).get_vector());
    EXPECT_TRUE(test::he::all_close(
        read_vector<double>(a),
        (test::NDArray<double, 2>({{1, 2}, {3, 4}, {5, 6}})).get_vector()));
  }
  {
    auto a = he_backend->create_packed_cipher_tensor(element::i64, shape);
    copy_data(a,
              test::NDArray<int64_t, 2>({{1, 2}, {3, 4}, {5, 6}}).get_vector());
    EXPECT_TRUE(test::he::all_close(
        read_vector<int64_t>(a),
        (test::NDArray<int64_t, 2>({{1, 2}, {3, 4}, {5, 6}})).get_vector()));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, cipher_tv_batch_write_read_2_3_complex) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());
  he_backend->update_encryption_parameters(
      HESealEncryptionParameters::default_complex_packing_parms());

  Shape shape{2, 3};
  {
    auto a = he_backend->create_packed_cipher_tensor(element::f32, shape);
    copy_data(a,
              test::NDArray<float, 2>({{1, 2}, {3, 4}, {5, 6}}).get_vector());
    EXPECT_TRUE(test::he::all_close(
        read_vector<float>(a),
        (test::NDArray<float, 2>({{1, 2}, {3, 4}, {5, 6}})).get_vector()));
  }
  {
    auto a = he_backend->create_packed_cipher_tensor(element::f64, shape);
    copy_data(a,
              test::NDArray<double, 2>({{1, 2}, {3, 4}, {5, 6}}).get_vector());
    EXPECT_TRUE(test::he::all_close(
        read_vector<double>(a),
        (test::NDArray<double, 2>({{1, 2}, {3, 4}, {5, 6}})).get_vector()));
  }
  {
    auto a = he_backend->create_packed_cipher_tensor(element::i64, shape);
    copy_data(a,
              test::NDArray<int64_t, 2>({{1, 2}, {3, 4}, {5, 6}}).get_vector());
    EXPECT_TRUE(test::he::all_close(
        read_vector<int64_t>(a),
        (test::NDArray<int64_t, 2>({{1, 2}, {3, 4}, {5, 6}})).get_vector()));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, cipher_tv_batch_write_read_2_1) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());

  Shape shape{2, 1};
  {
    auto a = he_backend->create_packed_cipher_tensor(element::f32, shape);
    copy_data(a, test::NDArray<float, 2>({{1, 2}}).get_vector());
    EXPECT_TRUE(
        test::he::all_close(read_vector<float>(a),
                            (test::NDArray<float, 2>({{1, 2}})).get_vector()));
  }
  {
    auto a = he_backend->create_packed_cipher_tensor(element::f64, shape);
    copy_data(a, test::NDArray<double, 2>({{1, 2}}).get_vector());
    EXPECT_TRUE(
        test::he::all_close(read_vector<double>(a),
                            (test::NDArray<double, 2>({{1, 2}})).get_vector()));
  }
  {
    auto a = he_backend->create_packed_cipher_tensor(element::i64, shape);
    copy_data(a, test::NDArray<int64_t, 2>({{1, 2}}).get_vector());
    EXPECT_TRUE(test::he::all_close(
        read_vector<int64_t>(a),
        (test::NDArray<int64_t, 2>({{1, 2}})).get_vector()));
  }
}
