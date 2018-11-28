//*****************************************************************************
// Copyright 2018 Intel Corporation
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

#include <assert.h>
#include <algorithm>

#include "seal/ckks/he_seal_ckks_backend.hpp"
#include "test_util.hpp"
#include "util/test_tools.hpp"

#include "ngraph/ngraph.hpp"
#include "ngraph/util.hpp"

#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "";

static void run_cryptonets_benchmark(string backend_name,
                                     size_t batch_size = 1) {
  if (backend_name == "INTERPRETER") {
    assert(batch_size == 1);
  }
  auto backend = runtime::Backend::create(backend_name);
  auto he_backend = dynamic_pointer_cast<runtime::he::HEBackend>(backend);
  if (he_backend) {
    he_backend->set_optimized_add(false);
    he_backend->set_optimized_mult(false);
  }

  vector<float> x = read_binary_constant(
      file_util::path_join(HE_SERIALIZED_ZOO, "x_test_4096.bin"),
      batch_size * 784);
  vector<float> y = read_binary_constant(
      file_util::path_join(HE_SERIALIZED_ZOO, "y_test_4096.bin"),
      batch_size * 10);

  // Global stop watch
  stopwatch sw_global;
  sw_global.start();

  // Load graph
  stopwatch sw_load_model;
  sw_load_model.start();
  const string filename = "mnist_cryptonets_batch_" + to_string(batch_size);
  const string json_path =
      file_util::path_join(HE_SERIALIZED_ZOO, filename + ".json");
  const string json_string = file_util::read_file_to_string(json_path);
  shared_ptr<Function> f = deserialize(json_string);
  NGRAPH_INFO << "Deserialize graph";
  NGRAPH_INFO << "x size " << x.size();
  NGRAPH_INFO << "Inputs loaded";
  sw_load_model.stop();
  NGRAPH_INFO << "sw_load_model: " << sw_load_model.get_milliseconds() << "ms";

  // Create input tensor and copy tensors; create output tensors
  stopwatch sw_encrypt_input;
  sw_encrypt_input.start();
  auto parameters = f->get_parameters();
  vector<shared_ptr<runtime::Tensor>> parameter_tvs;
  for (const auto& parameter : parameters) {
    auto& shape = parameter->get_shape();
    auto& type = parameter->get_element_type();

    std::shared_ptr<runtime::Tensor> parameter_cipher_tv =
        backend->create_tensor(type, shape);

    NGRAPH_INFO << "Creating input shape: " << join(shape, "x");

    if (shape == Shape{batch_size, 784}) {
      NGRAPH_INFO << "Copying " << shape_size(shape) << " elements";

      copy_data(parameter_cipher_tv, x);
      parameter_tvs.push_back(parameter_cipher_tv);
    } else {
      NGRAPH_INFO << "Invalid shape" << join(shape, "x");
      throw ngraph_error("Invalid shape " + shape_size(shape));
    }
  }

  auto results = f->get_results();
  vector<shared_ptr<runtime::Tensor>> result_tvs;
  for (const auto& result : results) {
    auto& shape = result->get_shape();
    auto& type = result->get_element_type();
    NGRAPH_INFO << "Creating output shape: " << join(shape, "x");

    result_tvs.push_back(backend->create_tensor(type, shape));
  }
  sw_encrypt_input.stop();
  NGRAPH_INFO << "sw_encrypt_input: " << sw_encrypt_input.get_milliseconds()
              << "ms";

  // Run model
  NGRAPH_INFO << "calling function";
  stopwatch sw_run_model;
  sw_run_model.start();
  backend->call(f, result_tvs, parameter_tvs);
  sw_run_model.stop();
  NGRAPH_INFO << "sw_run_model: " << sw_run_model.get_milliseconds() << "ms";

  // Decrypt output
  stopwatch sw_decrypt_output;
  sw_decrypt_output.start();
  auto result = (backend_name == "INTERPRETER")
                    ? read_vector<float>(result_tvs[0])
                    : generalized_read_vector<float>(result_tvs[0]);
  sw_decrypt_output.stop();
  NGRAPH_INFO << "sw_decrypt_output: " << sw_decrypt_output.get_milliseconds()
              << "ms";

  // Stop global stop watch
  sw_global.stop();
  NGRAPH_INFO << "sw_global: " << sw_global.get_milliseconds() << "ms";

  // Check prediction vs ground truth
  vector<int> y_gt_label = batched_argmax(y);
  vector<int> y_predicted_label = batched_argmax(result);

  size_t error_count = 0;
  for (size_t i = 0; i < y_gt_label.size(); ++i) {
    if (y_gt_label[i] != y_predicted_label[i]) {
      // NGRAPH_INFO << "index " << i << " y_gt_label != y_predicted_label: " <<
      // y_gt_label[i]
      //             << " != " << y_predicted_label[i];
      error_count++;
    }
  }
  NGRAPH_INFO << "Error count " << error_count << " of " << batch_size
              << " elements.";
  float accuracy = 1.f - (float)(error_count) / batch_size;
  NGRAPH_INFO << "Accuracy: " << accuracy;

  assert(accuracy > 0.97);

  // Print results
  NGRAPH_INFO << "[Summary]";
  NGRAPH_INFO << "sw_load_model: " << sw_load_model.get_milliseconds() << "ms";
  NGRAPH_INFO << "sw_encrypt_input: " << sw_encrypt_input.get_milliseconds()
              << "ms";
  NGRAPH_INFO << "sw_run_model: " << sw_run_model.get_milliseconds() << "ms";
  NGRAPH_INFO << "sw_decrypt_output: " << sw_decrypt_output.get_milliseconds()
              << "ms";
  NGRAPH_INFO << "sw_global: " << sw_global.get_milliseconds() << "ms";
};

NGRAPH_TEST(Cryptonets, CKKS_1) { run_cryptonets_benchmark("HE:SEAL:CKKS", 1); }

NGRAPH_TEST(Cryptonets, CKKS_2) { run_cryptonets_benchmark("HE:SEAL:CKKS", 2); }

NGRAPH_TEST(Cryptonets, CKKS_4) { run_cryptonets_benchmark("HE:SEAL:CKKS", 4); }

NGRAPH_TEST(Cryptonets, CKKS_8) { run_cryptonets_benchmark("HE:SEAL:CKKS", 8); }

NGRAPH_TEST(Cryptonets, CKKS_16) {
  run_cryptonets_benchmark("HE:SEAL:CKKS", 16);
}

NGRAPH_TEST(Cryptonets, CKKS_32) {
  run_cryptonets_benchmark("HE:SEAL:CKKS", 32);
}

NGRAPH_TEST(Cryptonets, CKKS_64) {
  run_cryptonets_benchmark("HE:SEAL:CKKS", 64);
}

NGRAPH_TEST(Cryptonets, CKKS_128) {
  run_cryptonets_benchmark("HE:SEAL:CKKS", 128);
}

NGRAPH_TEST(Cryptonets, CKKS_256) {
  run_cryptonets_benchmark("HE:SEAL:CKKS", 256);
}

NGRAPH_TEST(Cryptonets, CKKS_512) {
  run_cryptonets_benchmark("HE:SEAL:CKKS", 512);
}

NGRAPH_TEST(Cryptonets, CKKS_1024) {
  run_cryptonets_benchmark("HE:SEAL:CKKS", 1024);
}

NGRAPH_TEST(Cryptonets, CKKS_2048) {
  run_cryptonets_benchmark("HE:SEAL:CKKS", 2048);
}

NGRAPH_TEST(Cryptonets, CKKS_4096) {
  run_cryptonets_benchmark("HE:SEAL:CKKS", 4096);
}
