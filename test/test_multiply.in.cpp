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

#include "ngraph/ngraph.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/multiply_seal.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "test_util.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, multiply_2_3) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  Shape shape{2, 3};
  {
    auto a = make_shared<op::Parameter>(element::f32, shape);
    auto b = make_shared<op::Parameter>(element::f32, shape);
    auto t = make_shared<op::Multiply>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a, b});
    auto tensors_list =
        generate_plain_cipher_tensors({t}, {a, b}, backend.get());
    for (auto tensors : tensors_list) {
      auto results = get<0>(tensors);
      auto inputs = get<1>(tensors);
      auto t_a = inputs[0];
      auto t_b = inputs[1];
      auto t_result = results[0];
      copy_data(t_a,
                test::NDArray<float, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector());
      copy_data(
          t_b, test::NDArray<float, 2>({{7, 8, 9}, {10, 11, 12}}).get_vector());
      auto handle = backend->compile(f);
      handle->call_with_validate({t_result}, {t_a, t_b});
      EXPECT_TRUE(all_close(
          read_vector<float>(t_result),
          (test::NDArray<float, 2>({{7, 16, 27}, {40, 55, 72}})).get_vector(),
          1e-3f));
    }
  }
  {
    auto a = make_shared<op::Parameter>(element::f64, shape);
    auto b = make_shared<op::Parameter>(element::f64, shape);
    auto t = make_shared<op::Multiply>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a, b});
    auto tensors_list =
        generate_plain_cipher_tensors({t}, {a, b}, backend.get());
    for (auto tensors : tensors_list) {
      auto results = get<0>(tensors);
      auto inputs = get<1>(tensors);
      auto t_a = inputs[0];
      auto t_b = inputs[1];
      auto t_result = results[0];
      copy_data(t_a,
                test::NDArray<double, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector());
      copy_data(
          t_b,
          test::NDArray<double, 2>({{7, 8, 9}, {10, 11, 12}}).get_vector());
      auto handle = backend->compile(f);
      handle->call_with_validate({t_result}, {t_a, t_b});
      EXPECT_TRUE(all_close(
          read_vector<double>(t_result),
          (test::NDArray<double, 2>({{7, 16, 27}, {40, 55, 72}})).get_vector(),
          1e-3));
    }
  }
  {
    auto a = make_shared<op::Parameter>(element::i64, shape);
    auto b = make_shared<op::Parameter>(element::i64, shape);
    auto t = make_shared<op::Multiply>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a, b});
    auto tensors_list =
        generate_plain_cipher_tensors({t}, {a, b}, backend.get());
    for (auto tensors : tensors_list) {
      auto results = get<0>(tensors);
      auto inputs = get<1>(tensors);
      auto t_a = inputs[0];
      auto t_b = inputs[1];
      auto t_result = results[0];
      copy_data(t_a,
                test::NDArray<int64_t, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector());
      copy_data(
          t_b,
          test::NDArray<int64_t, 2>({{7, 8, 9}, {10, 11, 12}}).get_vector());
      auto handle = backend->compile(f);
      handle->call_with_validate({t_result}, {t_a, t_b});
      EXPECT_TRUE(all_close(
          read_vector<int64_t>(t_result),
          (test::NDArray<int64_t, 2>({{7, 16, 27}, {40, 55, 72}})).get_vector(),
          0L));
    }
  }
}

NGRAPH_TEST(${BACKEND_NAME}, multiply_2_3_halves) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  Shape shape{2, 3};
  {
    auto a = make_shared<op::Parameter>(element::f32, shape);
    auto b = make_shared<op::Parameter>(element::f32, shape);
    auto t = make_shared<op::Multiply>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a, b});
    // Create some tensors for input/output
    auto tensors_list =
        generate_plain_cipher_tensors({t}, {a, b}, backend.get());
    for (auto tensors : tensors_list) {
      auto results = get<0>(tensors);
      auto inputs = get<1>(tensors);
      auto t_a = inputs[0];
      auto t_b = inputs[1];
      auto t_result = results[0];
      copy_data(t_a, test::NDArray<float, 2>({{1.5, 2.5, 3.5}, {4.5, 5.5, 6.5}})
                         .get_vector());
      copy_data(t_b,
                test::NDArray<float, 2>({{7.5, 8.5, 9.5}, {10.5, 11.5, 12.5}})
                    .get_vector());
      auto handle = backend->compile(f);
      handle->call_with_validate({t_result}, {t_a, t_b});
      EXPECT_TRUE(all_close(read_vector<float>(t_result),
                            (test::NDArray<float, 2>({{11.25, 21.25, 33.25},
                                                      {47.25, 63.25, 81.25}}))
                                .get_vector(),
                            1e-3f));
    }
  }
  {
    auto a = make_shared<op::Parameter>(element::f64, shape);
    auto b = make_shared<op::Parameter>(element::f64, shape);
    auto t = make_shared<op::Multiply>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a, b});
    // Create some tensors for input/output
    auto tensors_list =
        generate_plain_cipher_tensors({t}, {a, b}, backend.get());
    for (auto tensors : tensors_list) {
      auto results = get<0>(tensors);
      auto inputs = get<1>(tensors);
      auto t_a = inputs[0];
      auto t_b = inputs[1];
      auto t_result = results[0];
      copy_data(t_a,
                test::NDArray<double, 2>({{1.5, 2.5, 3.5}, {4.5, 5.5, 6.5}})
                    .get_vector());
      copy_data(t_b,
                test::NDArray<double, 2>({{7.5, 8.5, 9.5}, {10.5, 11.5, 12.5}})
                    .get_vector());
      auto handle = backend->compile(f);
      handle->call_with_validate({t_result}, {t_a, t_b});
      EXPECT_TRUE(all_close(read_vector<double>(t_result),
                            (test::NDArray<double, 2>({{11.25, 21.25, 33.25},
                                                       {47.25, 63.25, 81.25}}))
                                .get_vector(),
                            1e-3));
    }
  }
}

NGRAPH_TEST(${BACKEND_NAME}, square_2_3) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  Shape shape{2, 3};
  {
    auto a = make_shared<op::Parameter>(element::f32, shape);
    auto t = make_shared<op::Multiply>(a, a);
    auto f = make_shared<Function>(t, ParameterVector{a});
    auto tensors_list =
        generate_plain_cipher_tensors({t}, {a}, backend.get(), true);
    for (auto tensors : tensors_list) {
      auto results = get<0>(tensors);
      auto inputs = get<1>(tensors);
      auto t_a = inputs[0];
      auto t_result = results[0];
      copy_data(t_a,
                test::NDArray<float, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector());
      auto handle = backend->compile(f);
      handle->call_with_validate({t_result}, {t_a});
      EXPECT_TRUE(all_close(
          read_vector<float>(t_result),
          (test::NDArray<float, 2>({{1, 4, 9}, {16, 25, 36}})).get_vector(),
          1e-3f));
    }
  }
  {
    auto a = make_shared<op::Parameter>(element::f64, shape);
    auto t = make_shared<op::Multiply>(a, a);
    auto f = make_shared<Function>(t, ParameterVector{a});
    auto tensors_list =
        generate_plain_cipher_tensors({t}, {a}, backend.get(), true);
    for (auto tensors : tensors_list) {
      auto results = get<0>(tensors);
      auto inputs = get<1>(tensors);
      auto t_a = inputs[0];
      auto t_result = results[0];
      copy_data(t_a,
                test::NDArray<double, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector());
      auto handle = backend->compile(f);
      handle->call_with_validate({t_result}, {t_a});
      EXPECT_TRUE(all_close(
          read_vector<double>(t_result),
          (test::NDArray<double, 2>({{1, 4, 9}, {16, 25, 36}})).get_vector(),
          1e-3));
    }
  }
  {
    auto a = make_shared<op::Parameter>(element::i64, shape);
    auto t = make_shared<op::Multiply>(a, a);
    auto f = make_shared<Function>(t, ParameterVector{a});
    auto tensors_list =
        generate_plain_cipher_tensors({t}, {a}, backend.get(), true);
    for (auto tensors : tensors_list) {
      auto results = get<0>(tensors);
      auto inputs = get<1>(tensors);
      auto t_a = inputs[0];
      auto t_result = results[0];
      copy_data(t_a,
                test::NDArray<int64_t, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector());
      auto handle = backend->compile(f);
      handle->call_with_validate({t_result}, {t_a});
      EXPECT_TRUE(all_close(
          read_vector<int64_t>(t_result),
          (test::NDArray<int64_t, 2>({{1, 4, 9}, {16, 25, 36}})).get_vector(),
          0L));
    }
  }
}

NGRAPH_TEST(${BACKEND_NAME}, multiply_optimized_2_3) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  Shape shape{2, 3};
  {
    auto a = make_shared<op::Parameter>(element::f32, shape);
    auto b = make_shared<op::Parameter>(element::f32, shape);
    auto t = make_shared<op::Multiply>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a, b});
    auto tensors_list =
        generate_plain_cipher_tensors({t}, {a, b}, backend.get());
    for (auto tensors : tensors_list) {
      auto results = get<0>(tensors);
      auto inputs = get<1>(tensors);
      auto t_a = inputs[0];
      auto t_b = inputs[1];
      auto t_result = results[0];
      copy_data(t_a,
                test::NDArray<float, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector());
      copy_data(t_b,
                test::NDArray<float, 2>({{-1, 0, 1}, {-1, 0, 1}}).get_vector());
      auto handle = backend->compile(f);
      handle->call_with_validate({t_result}, {t_a, t_b});
      EXPECT_TRUE(all_close(
          read_vector<float>(t_result),
          (test::NDArray<float, 2>({{-1, 0, 3}, {-4, 0, 6}})).get_vector(),
          1e-3f));
    }
  }
  {
    auto a = make_shared<op::Parameter>(element::f64, shape);
    auto b = make_shared<op::Parameter>(element::f64, shape);
    auto t = make_shared<op::Multiply>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a, b});
    auto tensors_list =
        generate_plain_cipher_tensors({t}, {a, b}, backend.get());
    for (auto tensors : tensors_list) {
      auto results = get<0>(tensors);
      auto inputs = get<1>(tensors);
      auto t_a = inputs[0];
      auto t_b = inputs[1];
      auto t_result = results[0];
      copy_data(t_a,
                test::NDArray<double, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector());
      copy_data(
          t_b, test::NDArray<double, 2>({{-1, 0, 1}, {-1, 0, 1}}).get_vector());
      auto handle = backend->compile(f);
      handle->call_with_validate({t_result}, {t_a, t_b});
      EXPECT_TRUE(all_close(
          read_vector<double>(t_result),
          (test::NDArray<double, 2>({{-1, 0, 3}, {-4, 0, 6}})).get_vector(),
          1e-3));
    }
  }
  {
    auto a = make_shared<op::Parameter>(element::i64, shape);
    auto b = make_shared<op::Parameter>(element::i64, shape);
    auto t = make_shared<op::Multiply>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a, b});
    auto tensors_list =
        generate_plain_cipher_tensors({t}, {a, b}, backend.get());
    for (auto tensors : tensors_list) {
      auto results = get<0>(tensors);
      auto inputs = get<1>(tensors);
      auto t_a = inputs[0];
      auto t_b = inputs[1];
      auto t_result = results[0];
      copy_data(t_a,
                test::NDArray<int64_t, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector());
      copy_data(
          t_b,
          test::NDArray<int64_t, 2>({{-1, 0, 1}, {-1, 0, 1}}).get_vector());
      auto handle = backend->compile(f);
      handle->call_with_validate({t_result}, {t_a, t_b});
      EXPECT_TRUE(all_close(
          read_vector<int64_t>(t_result),
          (test::NDArray<int64_t, 2>({{-1, 0, 3}, {-4, 0, 6}})).get_vector(),
          0L));
    }
  }
}

NGRAPH_TEST(${BACKEND_NAME}, multiply_4_3_batch) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());
  Shape shape_a{4, 3};
  Shape shape_b{4, 3};
  Shape shape_r{4, 3};
  {
    auto a = make_shared<op::Parameter>(element::f32, shape_a);
    auto b = make_shared<op::Parameter>(element::f32, shape_b);
    auto t = make_shared<op::Multiply>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a, b});
    auto t_a = he_backend->create_packed_plain_tensor(element::f32, shape_a);
    auto t_b = he_backend->create_packed_plain_tensor(element::f32, shape_b);
    auto t_result =
        he_backend->create_packed_plain_tensor(element::f32, shape_r);
    copy_data(t_a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    copy_data(t_b,
              vector<float>{13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a, t_b});
    EXPECT_TRUE(all_close(
        (vector<float>{13, 28, 45, 64, 85, 108, 133, 160, 189, 220, 253, 288}),
        read_vector<float>(t_result), 1e-3f));
  }
  {
    auto a = make_shared<op::Parameter>(element::f64, shape_a);
    auto b = make_shared<op::Parameter>(element::f64, shape_b);
    auto t = make_shared<op::Multiply>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a, b});
    auto t_a = he_backend->create_packed_plain_tensor(element::f64, shape_a);
    auto t_b = he_backend->create_packed_plain_tensor(element::f64, shape_b);
    auto t_result =
        he_backend->create_packed_plain_tensor(element::f64, shape_r);
    copy_data(t_a, vector<double>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    copy_data(t_b,
              vector<double>{13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a, t_b});
    EXPECT_TRUE(all_close(
        (vector<double>{13, 28, 45, 64, 85, 108, 133, 160, 189, 220, 253, 288}),
        read_vector<double>(t_result), 1e-3));
  }
  {
    auto a = make_shared<op::Parameter>(element::i64, shape_a);
    auto b = make_shared<op::Parameter>(element::i64, shape_b);
    auto t = make_shared<op::Multiply>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a, b});
    auto t_a = he_backend->create_packed_plain_tensor(element::i64, shape_a);
    auto t_b = he_backend->create_packed_plain_tensor(element::i64, shape_b);
    auto t_result =
        he_backend->create_packed_plain_tensor(element::i64, shape_r);
    copy_data(t_a, vector<int64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    copy_data(t_b,
              vector<int64_t>{13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a, t_b});
    EXPECT_TRUE(all_close((vector<int64_t>{13, 28, 45, 64, 85, 108, 133, 160,
                                           189, 220, 253, 288}),
                          read_vector<int64_t>(t_result), 0L));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, multiply_2_3_plain_complex) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());
  he_backend->complex_packing() = true;
  Shape shape{2, 3};
  {
    auto a = make_shared<op::Parameter>(element::f32, shape);
    auto b = make_shared<op::Parameter>(element::f32, shape);
    auto t = make_shared<op::Multiply>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a, b});
    auto t_a = he_backend->create_plain_tensor(element::f32, shape);
    auto t_b = he_backend->create_plain_tensor(element::f32, shape);
    auto t_result = he_backend->create_plain_tensor(element::f32, shape);
    copy_data(t_a, vector<float>{1, 2, 3, 4, 5, 6});
    copy_data(t_b, vector<float>{7, 8, 9, 10, 11, 12});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a, t_b});
    EXPECT_TRUE(all_close((vector<float>{7, 16, 27, 40, 55, 72}),
                          read_vector<float>(t_result), 1e-3f));
  }
  {
    auto a = make_shared<op::Parameter>(element::f64, shape);
    auto b = make_shared<op::Parameter>(element::f64, shape);
    auto t = make_shared<op::Multiply>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a, b});
    auto t_a = he_backend->create_plain_tensor(element::f64, shape);
    auto t_b = he_backend->create_plain_tensor(element::f64, shape);
    auto t_result = he_backend->create_plain_tensor(element::f64, shape);
    copy_data(t_a, vector<double>{1, 2, 3, 4, 5, 6});
    copy_data(t_b, vector<double>{7, 8, 9, 10, 11, 12});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a, t_b});
    EXPECT_TRUE(all_close((vector<double>{7, 16, 27, 40, 55, 72}),
                          read_vector<double>(t_result), 1e-3));
  }
  {
    auto a = make_shared<op::Parameter>(element::i64, shape);
    auto b = make_shared<op::Parameter>(element::i64, shape);
    auto t = make_shared<op::Multiply>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a, b});
    auto t_a = he_backend->create_plain_tensor(element::i64, shape);
    auto t_b = he_backend->create_plain_tensor(element::i64, shape);
    auto t_result = he_backend->create_plain_tensor(element::i64, shape);
    copy_data(t_a, vector<int64_t>{1, 2, 3, 4, 5, 6});
    copy_data(t_b, vector<int64_t>{7, 8, 9, 10, 11, 12});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a, t_b});
    EXPECT_TRUE(all_close((vector<int64_t>{7, 16, 27, 40, 55, 72}),
                          read_vector<int64_t>(t_result), 0L));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, multiply_2_3_cipher_plain_complex) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());
  he_backend->complex_packing() = true;
  Shape shape{2, 3};
  {
    auto a = make_shared<op::Parameter>(element::f32, shape);
    auto b = make_shared<op::Parameter>(element::f32, shape);
    auto t = make_shared<op::Multiply>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a, b});
    auto t_a = he_backend->create_cipher_tensor(element::f32, shape);
    auto t_b = he_backend->create_plain_tensor(element::f32, shape);
    auto t_result = he_backend->create_cipher_tensor(element::f32, shape);
    copy_data(t_a, vector<float>{-2, -1, 0, 1, 2, 3});
    copy_data(t_b, vector<float>{3, -2, 5, 3, 2, -5});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a, t_b});
    EXPECT_TRUE(all_close((vector<float>{-6, 2, 0, 3, 4, -15}),
                          read_vector<float>(t_result), 1e-3f));
  }
  {
    auto a = make_shared<op::Parameter>(element::f64, shape);
    auto b = make_shared<op::Parameter>(element::f64, shape);
    auto t = make_shared<op::Multiply>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a, b});
    auto t_a = he_backend->create_cipher_tensor(element::f64, shape);
    auto t_b = he_backend->create_plain_tensor(element::f64, shape);
    auto t_result = he_backend->create_cipher_tensor(element::f64, shape);
    copy_data(t_a, vector<double>{-2, -1, 0, 1, 2, 3});
    copy_data(t_b, vector<double>{3, -2, 5, 3, 2, -5});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a, t_b});
    EXPECT_TRUE(all_close((vector<double>{-6, 2, 0, 3, 4, -15}),
                          read_vector<double>(t_result), 1e-3));
  }
  {
    auto a = make_shared<op::Parameter>(element::i64, shape);
    auto b = make_shared<op::Parameter>(element::i64, shape);
    auto t = make_shared<op::Multiply>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a, b});
    auto t_a = he_backend->create_cipher_tensor(element::i64, shape);
    auto t_b = he_backend->create_plain_tensor(element::i64, shape);
    auto t_result = he_backend->create_cipher_tensor(element::i64, shape);
    copy_data(t_a, vector<int64_t>{-2, -1, 0, 1, 2, 3});
    copy_data(t_b, vector<int64_t>{3, -2, 5, 3, 2, -5});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a, t_b});
    EXPECT_TRUE(all_close((vector<int64_t>{-6, 2, 0, 3, 4, -15}),
                          read_vector<int64_t>(t_result), 0L));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, multiply_2_3_cipher_plain) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());
  Shape shape{2, 3};
  {
    auto a = make_shared<op::Parameter>(element::f32, shape);
    auto b = make_shared<op::Parameter>(element::f32, shape);
    auto t = make_shared<op::Multiply>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a, b});
    auto t_a = he_backend->create_cipher_tensor(element::f32, shape);
    auto t_b = he_backend->create_plain_tensor(element::f32, shape);
    auto t_result = he_backend->create_cipher_tensor(element::f32, shape);
    copy_data(t_a, vector<float>{-2, -1, 0, 1, 2, 3});
    copy_data(t_b, vector<float>{3, -2, 5, 3, 2, -5});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a, t_b});
    EXPECT_TRUE(all_close((vector<float>{-6, 2, 0, 3, 4, -15}),
                          read_vector<float>(t_result), 1e-3f));
  }
  {
    auto a = make_shared<op::Parameter>(element::f64, shape);
    auto b = make_shared<op::Parameter>(element::f64, shape);
    auto t = make_shared<op::Multiply>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a, b});
    auto t_a = he_backend->create_cipher_tensor(element::f64, shape);
    auto t_b = he_backend->create_plain_tensor(element::f64, shape);
    auto t_result = he_backend->create_cipher_tensor(element::f64, shape);
    copy_data(t_a, vector<double>{-2, -1, 0, 1, 2, 3});
    copy_data(t_b, vector<double>{3, -2, 5, 3, 2, -5});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a, t_b});
    EXPECT_TRUE(all_close((vector<double>{-6, 2, 0, 3, 4, -15}),
                          read_vector<double>(t_result), 1e-3));
  }
  {
    auto a = make_shared<op::Parameter>(element::i64, shape);
    auto b = make_shared<op::Parameter>(element::i64, shape);
    auto t = make_shared<op::Multiply>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a, b});
    auto t_a = he_backend->create_cipher_tensor(element::i64, shape);
    auto t_b = he_backend->create_plain_tensor(element::i64, shape);
    auto t_result = he_backend->create_cipher_tensor(element::i64, shape);
    copy_data(t_a, vector<int64_t>{-2, -1, 0, 1, 2, 3});
    copy_data(t_b, vector<int64_t>{3, -2, 5, 3, 2, -5});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a, t_b});
    EXPECT_TRUE(all_close((vector<int64_t>{-6, 2, 0, 3, 4, -15}),
                          read_vector<int64_t>(t_result), 0L));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, multiply_2_3_cipher_cipher_complex) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());
  he_backend->complex_packing() = true;

  auto a = std::make_shared<ngraph::he::SealCiphertextWrapper>();
  auto b = std::make_shared<ngraph::he::SealCiphertextWrapper>();
  auto c = std::make_shared<ngraph::he::SealCiphertextWrapper>();
  auto out = std::make_shared<ngraph::he::SealCiphertextWrapper>();
  a->complex_packing() = true;
  b->complex_packing() = true;
  {
    // arg1 complex packing
    EXPECT_THROW(
        { scalar_multiply_seal(*a, *c, out, element::f32, *he_backend); },
        CheckFailure);
    // arg2 complex packing
    EXPECT_THROW(
        { scalar_multiply_seal(*c, *a, out, element::f32, *he_backend); },
        CheckFailure);
    // both args complex packing
    EXPECT_THROW(
        { scalar_multiply_seal(*a, *b, out, element::f32, *he_backend); },
        CheckFailure);
  }
  {
    // arg1 complex packing
    EXPECT_THROW(
        { scalar_multiply_seal(*a, *c, out, element::f64, *he_backend); },
        CheckFailure);
    // arg2 complex packing
    EXPECT_THROW(
        { scalar_multiply_seal(*c, *a, out, element::f64, *he_backend); },
        CheckFailure);
    // both args complex packing
    EXPECT_THROW(
        { scalar_multiply_seal(*a, *b, out, element::f64, *he_backend); },
        CheckFailure);
  }
}

NGRAPH_TEST(${BACKEND_NAME}, multiply_2_3_cipher_cipher_complex_noerror) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());
  he_backend->complex_packing() = true;
  Shape shape{2, 3};
  {
    auto a = make_shared<op::Parameter>(element::f32, shape);
    auto b = make_shared<op::Parameter>(element::f32, shape);
    auto t = make_shared<op::Multiply>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a, b});
    auto t_a = he_backend->create_packed_cipher_tensor(element::f32, shape);
    auto t_b = he_backend->create_packed_cipher_tensor(element::f32, shape);
    auto t_result =
        he_backend->create_packed_cipher_tensor(element::f32, shape);
    copy_data(t_a, vector<float>{-2, -1, 0, 1, 2, 3});
    copy_data(t_b, vector<float>{3, -2, 5, 3, 2, -5});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a, t_b});
    EXPECT_TRUE(all_close((vector<float>{-6, 2, 0, 3, 4, -15}),
                          read_vector<float>(t_result), 1e-2f));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, multiply_2_1_cipher_cipher_complex_noerror) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());
  he_backend->complex_packing() = true;
  Shape shape{2, 1};
  {
    auto a = make_shared<op::Parameter>(element::f32, shape);
    auto b = make_shared<op::Parameter>(element::f32, shape);
    auto t = make_shared<op::Multiply>(a, b);
    auto f = make_shared<Function>(t, ParameterVector{a, b});
    auto t_a = he_backend->create_packed_cipher_tensor(element::f32, shape);
    auto t_b = he_backend->create_packed_cipher_tensor(element::f32, shape);
    auto t_result =
        he_backend->create_packed_cipher_tensor(element::f32, shape);
    copy_data(t_a, vector<float>{-2, -1});
    copy_data(t_b, vector<float>{3, -2});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a, t_b});
    EXPECT_TRUE(
        all_close((vector<float>{-6, 2}), read_vector<float>(t_result), 1e-2f));
  }
}