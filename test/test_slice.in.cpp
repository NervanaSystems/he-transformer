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

#include "he_op_annotations.hpp"
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

NGRAPH_TEST(${BACKEND_NAME}, slice_scalar) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  Shape shape_a{};
  Shape shape_r{};

  {
    auto a = make_shared<op::Parameter>(element::f32, shape_a);
    auto r = make_shared<op::Slice>(a, Coordinate{}, Coordinate{});
    auto f = make_shared<Function>(r, ParameterVector{a});

    a->set_op_annotations(
        HEOpAnnotations::server_plaintext_packed_annotation());

    auto t_a =
        he_backend->create_plain_tensor(a->get_element_type(), a->get_shape());
    auto t_result =
        he_backend->create_plain_tensor(r->get_element_type(), r->get_shape());

    copy_data(t_a, vector<float>{312});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a});
    EXPECT_EQ((vector<float>{312}), read_vector<float>(t_result));
  }

  {
    auto a = make_shared<op::Parameter>(element::f32, shape_a);
    auto r = make_shared<op::Slice>(a, Coordinate{}, Coordinate{});
    auto f = make_shared<Function>(r, ParameterVector{a});

    a->set_op_annotations(
        HEOpAnnotations::server_ciphertext_packed_annotation());

    auto t_a =
        he_backend->create_cipher_tensor(a->get_element_type(), a->get_shape());
    auto t_result =
        he_backend->create_cipher_tensor(r->get_element_type(), r->get_shape());

    copy_data(t_a, vector<float>{312});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a});
    EXPECT_EQ((vector<float>{312}), read_vector<float>(t_result));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, slice_matrix) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  Shape shape_a{4, 4};
  Shape shape_r{3, 2};

  {
    auto a = make_shared<op::Parameter>(element::f32, shape_a);
    auto r = make_shared<op::Slice>(a, Coordinate{0, 1}, Coordinate{3, 3});
    auto f = make_shared<Function>(r, ParameterVector{a});

    a->set_op_annotations(
        HEOpAnnotations::server_plaintext_unpacked_annotation());

    auto t_a =
        he_backend->create_plain_tensor(a->get_element_type(), a->get_shape());
    auto t_result =
        he_backend->create_plain_tensor(r->get_element_type(), r->get_shape());

    copy_data(t_a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                 15, 16});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a});
    EXPECT_TRUE(all_close((vector<float>{2, 3, 6, 7, 10, 11}),
                          read_vector<float>(t_result)));
  }
  {
    auto a = make_shared<op::Parameter>(element::f32, shape_a);
    auto r = make_shared<op::Slice>(a, Coordinate{0, 1}, Coordinate{3, 3});
    auto f = make_shared<Function>(r, ParameterVector{a});

    a->set_op_annotations(
        HEOpAnnotations::server_ciphertext_unpacked_annotation());

    auto t_a =
        he_backend->create_cipher_tensor(a->get_element_type(), a->get_shape());
    auto t_result =
        he_backend->create_cipher_tensor(r->get_element_type(), r->get_shape());

    copy_data(t_a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                 15, 16});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a});
    EXPECT_TRUE(all_close((vector<float>{2, 3, 6, 7, 10, 11}),
                          read_vector<float>(t_result)));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, slice_vector) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  Shape shape_a{16};
  Shape shape_r{12};

  {
    auto a = make_shared<op::Parameter>(element::f32, shape_a);
    auto r = make_shared<op::Slice>(a, Coordinate{2}, Coordinate{14});
    auto f = make_shared<Function>(r, ParameterVector{a});

    a->set_op_annotations(
        HEOpAnnotations::server_plaintext_unpacked_annotation());

    auto t_a =
        he_backend->create_plain_tensor(a->get_element_type(), a->get_shape());
    auto t_result =
        he_backend->create_plain_tensor(r->get_element_type(), r->get_shape());

    copy_data(t_a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                 14, 15});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a});
    EXPECT_TRUE(
        all_close((vector<float>{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}),
                  read_vector<float>(t_result)));
  }
  {
    auto a = make_shared<op::Parameter>(element::f32, shape_a);
    auto r = make_shared<op::Slice>(a, Coordinate{2}, Coordinate{14});
    auto f = make_shared<Function>(r, ParameterVector{a});

    a->set_op_annotations(
        HEOpAnnotations::server_ciphertext_unpacked_annotation());

    auto t_a =
        he_backend->create_cipher_tensor(a->get_element_type(), a->get_shape());
    auto t_result =
        he_backend->create_cipher_tensor(r->get_element_type(), r->get_shape());

    copy_data(t_a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                 14, 15});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a});
    EXPECT_TRUE(
        all_close((vector<float>{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}),
                  read_vector<float>(t_result)));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, slice_matrix_strided) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  Shape shape_a{4, 4};
  Shape shape_r{2, 2};

  {
    auto a = make_shared<op::Parameter>(element::f32, shape_a);
    auto r = make_shared<op::Slice>(a, Coordinate{1, 0}, Coordinate{4, 4},
                                    Strides{2, 3});
    auto f = make_shared<Function>(r, ParameterVector{a});

    a->set_op_annotations(
        HEOpAnnotations::server_plaintext_unpacked_annotation());

    auto t_a =
        he_backend->create_plain_tensor(a->get_element_type(), a->get_shape());
    auto t_result =
        he_backend->create_plain_tensor(r->get_element_type(), r->get_shape());

    copy_data(t_a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                 14, 15});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a});
    EXPECT_TRUE(
        all_close((vector<float>{4, 7, 12, 15}), read_vector<float>(t_result)));
  }
  {
    auto a = make_shared<op::Parameter>(element::f32, shape_a);
    auto r = make_shared<op::Slice>(a, Coordinate{1, 0}, Coordinate{4, 4},
                                    Strides{2, 3});
    auto f = make_shared<Function>(r, ParameterVector{a});

    a->set_op_annotations(
        HEOpAnnotations::server_ciphertext_unpacked_annotation());

    auto t_a =
        he_backend->create_cipher_tensor(a->get_element_type(), a->get_shape());
    auto t_result =
        he_backend->create_cipher_tensor(r->get_element_type(), r->get_shape());

    copy_data(t_a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                 14, 15});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a});
    EXPECT_TRUE(
        all_close((vector<float>{4, 7, 12, 15}), read_vector<float>(t_result)));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, slice_3d) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  Shape shape_a{4, 4, 4};
  Shape shape_r{2, 2, 2};

  {
    auto a = make_shared<op::Parameter>(element::f32, shape_a);
    auto r =
        make_shared<op::Slice>(a, Coordinate{1, 1, 1}, Coordinate{3, 3, 3});
    auto f = make_shared<Function>(r, ParameterVector{a});

    a->set_op_annotations(
        HEOpAnnotations::server_plaintext_unpacked_annotation());

    auto t_a =
        he_backend->create_plain_tensor(a->get_element_type(), a->get_shape());
    auto t_result =
        he_backend->create_plain_tensor(r->get_element_type(), r->get_shape());

    copy_data(t_a,
              vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                            26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                            39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
                            52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a});
    EXPECT_TRUE(all_close((vector<float>{21, 22, 25, 26, 37, 38, 41, 42}),
                          read_vector<float>(t_result)));
  }
  {
    auto a = make_shared<op::Parameter>(element::f32, shape_a);
    auto r =
        make_shared<op::Slice>(a, Coordinate{1, 1, 1}, Coordinate{3, 3, 3});
    auto f = make_shared<Function>(r, ParameterVector{a});

    a->set_op_annotations(
        HEOpAnnotations::server_ciphertext_unpacked_annotation());

    auto t_a =
        he_backend->create_cipher_tensor(a->get_element_type(), a->get_shape());
    auto t_result =
        he_backend->create_cipher_tensor(r->get_element_type(), r->get_shape());

    copy_data(t_a,
              vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                            26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                            39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
                            52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a});
    EXPECT_TRUE(all_close((vector<float>{21, 22, 25, 26, 37, 38, 41, 42}),
                          read_vector<float>(t_result)));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, slice_3d_strided) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  Shape shape_a{4, 4, 4};
  Shape shape_r{2, 2, 2};

  {
    auto a = make_shared<op::Parameter>(element::f32, shape_a);
    auto r = make_shared<op::Slice>(a, Coordinate{0, 0, 0}, Coordinate{4, 4, 4},
                                    Strides{2, 2, 2});
    auto f = make_shared<Function>(r, ParameterVector{a});

    a->set_op_annotations(
        HEOpAnnotations::server_plaintext_unpacked_annotation());

    auto t_a =
        he_backend->create_plain_tensor(a->get_element_type(), a->get_shape());
    auto t_result =
        he_backend->create_plain_tensor(r->get_element_type(), r->get_shape());

    copy_data(t_a,
              vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                            27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                            40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                            53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a});
    EXPECT_TRUE(all_close((vector<float>{1, 3, 9, 11, 33, 35, 41, 43}),
                          read_vector<float>(t_result)));
  }
  {
    auto a = make_shared<op::Parameter>(element::f32, shape_a);
    auto r = make_shared<op::Slice>(a, Coordinate{0, 0, 0}, Coordinate{4, 4, 4},
                                    Strides{2, 2, 2});

    auto f = make_shared<Function>(r, ParameterVector{a});

    a->set_op_annotations(
        HEOpAnnotations::server_ciphertext_unpacked_annotation());

    auto t_a =
        he_backend->create_cipher_tensor(a->get_element_type(), a->get_shape());
    auto t_result =
        he_backend->create_cipher_tensor(r->get_element_type(), r->get_shape());

    copy_data(t_a,
              vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                            27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                            40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                            53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a});
    EXPECT_TRUE(all_close((vector<float>{1, 3, 9, 11, 33, 35, 41, 43}),
                          read_vector<float>(t_result)));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, slice_3d_strided_different_strides) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  Shape shape_a{4, 4, 4};
  Shape shape_r{2, 2, 2};

  {
    auto a = make_shared<op::Parameter>(element::f32, shape_a);
    auto r = make_shared<op::Slice>(a, Coordinate{0, 0, 0}, Coordinate{4, 4, 4},
                                    Strides{2, 2, 3});
    auto f = make_shared<Function>(r, ParameterVector{a});

    a->set_op_annotations(
        HEOpAnnotations::server_plaintext_unpacked_annotation());

    auto t_a =
        he_backend->create_plain_tensor(a->get_element_type(), a->get_shape());
    auto t_result =
        he_backend->create_plain_tensor(r->get_element_type(), r->get_shape());

    copy_data(t_a,
              vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                            27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                            40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                            53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a});
    EXPECT_TRUE(all_close((vector<float>{1, 4, 9, 12, 33, 36, 41, 44}),
                          read_vector<float>(t_result)));
  }
  {
    auto a = make_shared<op::Parameter>(element::f32, shape_a);
    auto r = make_shared<op::Slice>(a, Coordinate{0, 0, 0}, Coordinate{4, 4, 4},
                                    Strides{2, 2, 3});
    auto f = make_shared<Function>(r, ParameterVector{a});

    a->set_op_annotations(
        HEOpAnnotations::server_ciphertext_unpacked_annotation());

    auto t_a =
        he_backend->create_cipher_tensor(a->get_element_type(), a->get_shape());
    auto t_result =
        he_backend->create_cipher_tensor(r->get_element_type(), r->get_shape());

    copy_data(t_a,
              vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                            14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                            27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                            40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                            53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64});
    auto handle = backend->compile(f);
    handle->call_with_validate({t_result}, {t_a});
    EXPECT_TRUE(all_close((vector<float>{1, 4, 9, 12, 33, 36, 41, 44}),
                          read_vector<float>(t_result)));
  }
}
