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

auto sub_test = [](const ngraph::Shape& shape, const bool arg1_encrypted,
                   const bool arg2_encrypted, const bool complex_packing,
                   const bool packed) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  if (complex_packing) {
    he_backend->update_encryption_parameters(
        ngraph::he::HESealEncryptionParameters::
            default_complex_packing_parms());
  }

  auto a = make_shared<op::Parameter>(element::f32, shape);
  auto b = make_shared<op::Parameter>(element::f32, shape);
  auto t = make_shared<op::Subtract>(a, b);
  auto f = make_shared<Function>(t, ParameterVector{a, b});

  auto annotation_from_flags = [](bool is_encrypted, bool is_packed) {
    if (is_encrypted && is_packed) {
      return HEOpAnnotations::server_ciphertext_packed_annotation();
    } else if (is_encrypted && !is_packed) {
      return HEOpAnnotations::server_ciphertext_unpacked_annotation();
    } else if (!is_encrypted && is_packed) {
      return HEOpAnnotations::server_plaintext_packed_annotation();
    } else if (!is_encrypted && !is_packed) {
      return HEOpAnnotations::server_ciphertext_unpacked_annotation();
    }
    throw ngraph_error("Logic error");
  };

  a->set_op_annotations(annotation_from_flags(arg1_encrypted, packed));
  b->set_op_annotations(annotation_from_flags(arg2_encrypted, packed));

  auto tensor_from_flags = [&](bool encrypted) {
    if (encrypted && packed) {
      return he_backend->create_packed_cipher_tensor(element::f32, shape);
    } else if (encrypted && !packed) {
      return he_backend->create_cipher_tensor(element::f32, shape);
    } else if (!encrypted && packed) {
      return he_backend->create_packed_plain_tensor(element::f32, shape);
    } else if (!encrypted && !packed) {
      return he_backend->create_plain_tensor(element::f32, shape);
    }
    throw ngraph_error("Logic error");
  };

  auto t_a = tensor_from_flags(arg1_encrypted);
  auto t_b = tensor_from_flags(arg2_encrypted);
  auto t_result = tensor_from_flags(arg1_encrypted | arg2_encrypted);

  std::vector<float> input_a;
  std::vector<float> input_b;
  std::vector<float> exp_result;

  for (int i = 0; i < ngraph::shape_size(shape); ++i) {
    input_a.emplace_back(i);

    if (i % 2 == 0) {
      input_b.emplace_back(i);
    } else {
      input_b.emplace_back(1 - i);
    }
    exp_result.emplace_back(input_a.back() - input_b.back());
  }
  copy_data(t_a, input_a);
  copy_data(t_b, input_b);

  auto handle = backend->compile(f);
  handle->call_with_validate({t_result}, {t_a, t_b});

  for (size_t i = 0; i < exp_result.size(); ++i) {
    NGRAPH_INFO << "Sub[ " << i << "]: (a=" << input_a[i]
                << " ) - (b=" << input_b[i] << ") => "
                << read_vector<float>(t_result)[i] << " (expected "
                << exp_result[i] << ")";
  }

  EXPECT_TRUE(all_close(read_vector<float>(t_result), exp_result, 1e-3f));
};

NGRAPH_TEST(${BACKEND_NAME}, sub_2_3_plain_plain_real_unpacked) {
  sub_test(ngraph::Shape{2, 3}, false, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, sub_2_3_plain_plain_real_packed) {
  sub_test(ngraph::Shape{2, 3}, false, false, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, sub_2_3_plain_plain_complex_unpacked) {
  sub_test(ngraph::Shape{2, 3}, false, false, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, sub_2_3_plain_plain_complex_packed) {
  sub_test(ngraph::Shape{2, 3}, false, false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, sub_2_3_plain_cipher_real_unpacked) {
  sub_test(ngraph::Shape{2, 3}, false, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, sub_2_3_plain_cipher_real_packed) {
  sub_test(ngraph::Shape{2, 3}, false, true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, sub_2_3_plain_cipher_complex_unpacked) {
  sub_test(ngraph::Shape{2, 3}, false, true, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, sub_2_3_plain_cipher_complex_packed) {
  sub_test(ngraph::Shape{2, 3}, false, true, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, sub_2_3_cipher_plain_real_unpacked) {
  sub_test(ngraph::Shape{2, 3}, true, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, sub_2_3_cipher_plain_real_packed) {
  sub_test(ngraph::Shape{2, 3}, true, false, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, sub_2_3_cipher_plain_complex_unpacked) {
  sub_test(ngraph::Shape{2, 3}, true, false, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, sub_2_3_cipher_plain_complex_packed) {
  sub_test(ngraph::Shape{2, 3}, true, false, true, true);
}

NGRAPH_TEST(${BACKEND_NAME}, sub_2_3_cipher_cipher_real_unpacked) {
  sub_test(ngraph::Shape{2, 3}, true, true, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, sub_2_3_cipher_cipher_real_packed) {
  sub_test(ngraph::Shape{2, 3}, true, true, false, true);
}

NGRAPH_TEST(${BACKEND_NAME}, sub_2_3_cipher_cipher_complex_unpacked) {
  sub_test(ngraph::Shape{2, 3}, true, true, true, false);
}

NGRAPH_TEST(${BACKEND_NAME}, sub_2_3_cipher_cipher_complex_packed) {
  sub_test(ngraph::Shape{2, 3}, true, true, true, true);
}
