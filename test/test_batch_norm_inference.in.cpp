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
#include "ngraph/pass/constant_folding.hpp"
#include "ngraph/pass/core_fusion.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "seal/he_seal_backend.hpp"
#include "test_util.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

static string s_manifest = "${MANIFEST}";

template <typename T>
class BatchNormInferenceTester {
 public:
  BatchNormInferenceTester(ngraph::he::HESealBackend* backend,
                           const Shape& input_shape, element::Type etype,
                           double epsilon)
      : m_he_backend(backend) {
    Shape channel_shape{input_shape.at(1)};

    auto Input = make_shared<op::Parameter>(etype, input_shape);
    auto Gamma = make_shared<op::Parameter>(etype, channel_shape);
    auto Beta = make_shared<op::Parameter>(etype, channel_shape);
    auto Mean = make_shared<op::Parameter>(etype, channel_shape);
    auto Variance = make_shared<op::Parameter>(etype, channel_shape);
    auto BN = make_shared<op::BatchNormInference>(Input, Gamma, Beta, Mean,
                                                  Variance, epsilon);
    m_function = make_shared<Function>(
        BN, ParameterVector{Input, Gamma, Beta, Mean, Variance});

    Input->set_op_annotations(
        HEOpAnnotations::server_ciphertext_unpacked_annotation());
    Gamma->set_op_annotations(
        HEOpAnnotations::server_plaintext_unpacked_annotation());
    Beta->set_op_annotations(
        HEOpAnnotations::server_plaintext_unpacked_annotation());
    Mean->set_op_annotations(
        HEOpAnnotations::server_plaintext_unpacked_annotation());
    Variance->set_op_annotations(
        HEOpAnnotations::server_plaintext_unpacked_annotation());

    m_input = backend->create_cipher_tensor(etype, input_shape);
    m_gamma = backend->create_plain_tensor(etype, channel_shape);
    m_beta = backend->create_plain_tensor(etype, channel_shape);
    m_mean = backend->create_plain_tensor(etype, channel_shape);
    m_variance = backend->create_plain_tensor(etype, channel_shape);
    m_normed_input = backend->create_cipher_tensor(etype, input_shape);
  }

  bool call(const vector<T>& input, const vector<T>& gamma,
            const vector<T>& beta, const vector<T>& mean,
            const vector<T>& variance, const vector<T>& normed_input) {
    copy_data(m_input, input);
    copy_data(m_gamma, gamma);
    copy_data(m_beta, beta);
    copy_data(m_mean, mean);
    copy_data(m_variance, variance);
    auto handle = m_he_backend->compile(m_function);
    handle->call_with_validate({m_normed_input},
                               {m_input, m_gamma, m_beta, m_mean, m_variance});
    auto res_normed_input = read_vector<T>(m_normed_input);
    return test::he::all_close(normed_input, res_normed_input);
  }

 protected:
  ngraph::he::HESealBackend* m_he_backend;
  shared_ptr<Function> m_function;
  shared_ptr<ngraph::runtime::Tensor> m_input;
  shared_ptr<ngraph::runtime::Tensor> m_gamma;
  shared_ptr<ngraph::runtime::Tensor> m_beta;
  shared_ptr<ngraph::runtime::Tensor> m_mean;
  shared_ptr<ngraph::runtime::Tensor> m_variance;
  shared_ptr<ngraph::runtime::Tensor> m_normed_input;
};

template <typename T>
class BatchNormInferenceTesterZeroEpsilon : public BatchNormInferenceTester<T> {
 public:
  // These are for documentation purposes only below
  using Input = test::NDArray<T, 2>;
  using Gamma = test::NDArray<T, 1>;
  using Beta = test::NDArray<T, 1>;
  using Mean = test::NDArray<T, 1>;
  using Variance = test::NDArray<T, 1>;
  using NormedInput = test::NDArray<T, 2>;

  BatchNormInferenceTesterZeroEpsilon(ngraph::he::HESealBackend* backend,
                                      element::Type etype)
      : BatchNormInferenceTester<T>(backend, Shape{2, 3}, etype, 0.0) {}

  bool test(const Input& input, const Gamma& gamma, const Beta& beta,
            const Mean& mean, const Variance& variance,
            const NormedInput& normed_input) {
    return BatchNormInferenceTester<T>::call(
        input.get_vector(), gamma.get_vector(), beta.get_vector(),
        mean.get_vector(), variance.get_vector(), normed_input.get_vector());
  }

  bool test_gamma() {
    return test(Input{{1.0, 2.0, 3.0}, {-1.0, -2.0, -3.0}},
                Gamma{2.0, 3.0, 4.0}, Beta{0.0, 0.0, 0.0}, Mean{0.0, 0.0, 0.0},
                Variance{1.0, 1.0, 1.0},
                NormedInput{{2.0, 6.0, 12.0}, {-2.0, -6.0, -12.0}});
  }

  bool test_beta() {
    return test(Input{{1.0, 2.0, 3.0}, {-1.0, -2.0, -3.0}},
                Gamma{1.0, 1.0, 1.0}, Beta{2.0, -2.0, 3.0}, Mean{0.0, 0.0, 0.0},
                Variance{1.0, 1.0, 1.0},
                NormedInput{{3.0, 0.0, 6.0}, {1.0, -4.0, 0.0}});
  }

  bool test_mean() {
    return test(Input{{1.0, 2.0, 3.0}, {-1.0, -2.0, -3.0}},
                Gamma{1.0, 1.0, 1.0}, Beta{0.0, 0.0, 0.0},
                Mean{-2.0, 2.0, -3.0}, Variance{1.0, 1.0, 1.0},
                NormedInput{{3.0, 0.0, 6.0}, {1.0, -4.0, 0.0}});
  }

  bool test_variance() {
    return test(Input{{1.0, 2.0, 3.0}, {-1.0, -2.0, -3.0}},
                Gamma{1.0, 1.0, 1.0}, Beta{0.0, 0.0, 0.0}, Mean{0.0, 0.0, 0.0},
                Variance{0.25, .0625, 4.0},
                NormedInput{{2.0, 8.0, 1.5}, {-2.0, -8.0, -1.5}});
  }
};

template <typename T>
class BatchNormInferenceTesterNonZeroEpsilon
    : public BatchNormInferenceTester<T> {
 public:
  // These are for documentation purposes only below
  using Input = test::NDArray<T, 2>;
  using Gamma = test::NDArray<T, 1>;
  using Beta = test::NDArray<T, 1>;
  using Mean = test::NDArray<T, 1>;
  using Variance = test::NDArray<T, 1>;
  using NormedInput = test::NDArray<T, 2>;

  BatchNormInferenceTesterNonZeroEpsilon(ngraph::he::HESealBackend* backend,
                                         element::Type etype)
      : BatchNormInferenceTester<T>(backend, Shape{2, 3}, etype, 0.25) {}

  bool test(const Input& input, const Gamma& gamma, const Beta& beta,
            const Mean& mean, const Variance& variance,
            const NormedInput& normed_input) {
    return BatchNormInferenceTester<T>::call(
        input.get_vector(), gamma.get_vector(), beta.get_vector(),
        mean.get_vector(), variance.get_vector(), normed_input.get_vector());
  }

  bool test_gamma() {
    return test(Input{{1.0, 2.0, 3.0}, {-1.0, -2.0, -3.0}},
                Gamma{2.0, 3.0, 4.0}, Beta{0.0, 0.0, 0.0}, Mean{0.0, 0.0, 0.0},
                Variance{0.75, 0.75, 0.75},
                NormedInput{{2.0, 6.0, 12.0}, {-2.0, -6.0, -12.0}});
  }

  bool test_beta() {
    return test(Input{{1.0, 2.0, 3.0}, {-1.0, -2.0, -3.0}},
                Gamma{1.0, 1.0, 1.0}, Beta{2.0, -2.0, 3.0}, Mean{0.0, 0.0, 0.0},
                Variance{0.75, 0.75, 0.75},
                NormedInput{{3.0, 0.0, 6.0}, {1.0, -4.0, 0.0}});
  }

  bool test_mean() {
    return test(Input{{1.0, 2.0, 3.0}, {-1.0, -2.0, -3.0}},
                Gamma{1.0, 1.0, 1.0}, Beta{0.0, 0.0, 0.0},
                Mean{-2.0, 2.0, -3.0}, Variance{0.75, 0.75, 0.75},
                NormedInput{{3.0, 0.0, 6.0}, {1.0, -4.0, 0.0}});
  }

  bool test_variance() {
    return test(Input{{3.0, 5.0, 1.0}, {-3.0, -5.0, -1.0}},
                Gamma{1.0, 1.0, 1.0}, Beta{0.0, 0.0, 0.0}, Mean{0.0, 0.0, 0.0},
                Variance{2.0, 6.0, 0.0},
                NormedInput{{2.0, 2.0, 2.0}, {-2.0, -2.0, -2.0}});
  }
};

NGRAPH_TEST(${BACKEND_NAME}, batch_norm_inference_0eps_f32) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  BatchNormInferenceTesterZeroEpsilon<float> bnt(he_backend, element::f32);
  EXPECT_TRUE(bnt.test_gamma()) << "Gamma test";
  EXPECT_TRUE(bnt.test_beta()) << "Beta test";
  EXPECT_TRUE(bnt.test_mean()) << "Mean test";
  EXPECT_TRUE(bnt.test_variance()) << "Variance test";
}

NGRAPH_TEST(${BACKEND_NAME}, batch_norm_inference_f32) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<ngraph::he::HESealBackend*>(backend.get());

  BatchNormInferenceTesterNonZeroEpsilon<float> bnt(he_backend, element::f32);
  EXPECT_TRUE(bnt.test_gamma()) << "Gamma test";
  EXPECT_TRUE(bnt.test_beta()) << "Beta test";
  EXPECT_TRUE(bnt.test_mean()) << "Mean test";
  EXPECT_TRUE(bnt.test_variance()) << "Variance test";
}

NGRAPH_TEST(${BACKEND_NAME}, batch_norm_fusion) {
  auto backend = runtime::Backend::create("INTERPRETER");

  Shape shape_input{1, 8, 3, 3};
  Shape shape_weights{2, 8, 1, 1};
  Shape shape_norm{2};

  vector<float> input{
      1.25f,  2.25f,  5.25f,  6.25f,  -1.25f, -1.25f, 3.25f, -4.25f, 7.25f,
      8.25f,  -1.25f, -1.25f, 1.25f,  2.25f,  -3.25f, 2.25f, 4.25f,  4.25f,
      1.25f,  2.25f,  -4.25f, 2.25f,  4.25f,  4.25f,  0.f,   0.f,    -1.f,
      0.f,    2.f,    2.f,    0.f,    0.f,    0.f,    0.f,   2.f,    2.f,
      1.25f,  2.25f,  5.25f,  6.25f,  1.25f,  1.25f,  3.25f, 4.25f,  -7.25f,
      8.25f,  1.25f,  -1.25f, -1.25f, 2.25f,  3.25f,  2.25f, -4.25f, -4.25f,
      -1.25f, -2.25f, 4.25f,  2.25f,  4.25f,  4.25f,  0.f,   0.f,    1.f,
      0.f,    -2.f,   2.f,    0.f,    0.f,    0.f,    0.f,   -2.f,   -2.f};

  vector<float> weight_vals{1.25f, 2.25f,  5.25f, 6.25f, -1.25f, -1.25f,
                            3.25f, -4.25f, 7.25f, 8.25f, -1.25f, 0.f,
                            0.f,   0.f,    0.f,   -2.f};

  vector<float> gamma_vals{-0.9384f, 0.01875f};
  vector<float> beta_vals{11.0f, 1.3f};
  vector<float> mean_vals{0.12f, 0.31f};
  vector<float> var_vals{0.01f, 0.11f};

  auto et = element::f32;

  auto make_function = [shape_input, shape_weights, shape_norm, gamma_vals,
                        weight_vals, beta_vals, mean_vals, var_vals, et]() {
    auto input_parm = make_shared<op::Parameter>(et, shape_input);
    auto weights = make_shared<op::Constant>(et, shape_weights, weight_vals);
    double eps = 0.001;
    auto gamma = make_shared<op::Constant>(et, shape_norm, gamma_vals);
    auto beta = make_shared<op::Constant>(et, shape_norm, beta_vals);
    auto mean = make_shared<op::Constant>(et, shape_norm, mean_vals);
    auto var = make_shared<op::Constant>(et, shape_norm, var_vals);
    auto conv = make_shared<op::Convolution>(input_parm, weights, Strides{1, 1},
                                             Strides{1, 1});
    auto bn =
        make_shared<op::BatchNormInference>(conv, gamma, beta, mean, var, eps);
    auto f = make_shared<Function>(NodeVector{bn}, ParameterVector{input_parm});
    return f;
  };

  auto orig_f = make_function();
  auto opt_f = make_function();

  pass::Manager pass_manager_opt;

  pass_manager_opt.register_pass<pass::CoreFusion>();
  pass_manager_opt.register_pass<pass::ConstantFolding>();
  pass_manager_opt.run_passes(opt_f);

  auto orig_ops = orig_f->get_ordered_ops();
  auto new_ops = opt_f->get_ordered_ops();

  auto t_orig_result = backend->create_tensor(element::f32, {1, 2, 3, 3});
  auto t_opt_result = backend->create_tensor(element::f32, {1, 2, 3, 3});
  auto t_input = backend->create_tensor(element::f32, shape_input);

  copy_data(t_input, input);

  auto orig_exec = backend->compile(orig_f);
  auto opt_exec = backend->compile(opt_f);

  orig_exec->call_with_validate({t_orig_result}, {t_input});
  opt_exec->call_with_validate({t_opt_result}, {t_input});

  EXPECT_TRUE(test::all_close(read_vector<float>(t_orig_result),
                              read_vector<float>(t_opt_result)));
}
