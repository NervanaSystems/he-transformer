
/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <algorithm>
#include <assert.h>

#include "ngraph/file_util.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"

#include "he_backend.hpp"
#include "he_heaan_backend.hpp"

using namespace std;
using namespace ngraph;

template <typename T>
void copy_data(std::shared_ptr<ngraph::runtime::TensorView> tv, const std::vector<T>& data)
{
    size_t data_size = data.size() * sizeof(T);
    tv->write(data.data(), 0, data_size);
}

template <typename T>
std::vector<T> read_vector(std::shared_ptr<ngraph::runtime::TensorView> tv)
{
    if (ngraph::element::from<T>() != tv->get_tensor_view_layout()->get_element_type())
    {
        throw std::invalid_argument("read_vector type must match TensorView type");
    }
    size_t element_count = ngraph::shape_size(tv->get_shape());
    size_t size = element_count * sizeof(T);
    std::vector<T> rc(element_count);
    tv->read(rc.data(), 0, size);
    return rc;
}

vector<float> read_binary_constant(const string filename, size_t num_elements)
{
    ifstream infile;
    vector<float> values(num_elements);
    infile.open(filename, ios::in | ios::binary);

    infile.read(reinterpret_cast<char*>(&values[0]), num_elements * sizeof(float));
    infile.close();
    return values;
}

int main()
{
    // Set batch_size = 1 for data loading
    // However, we don't create batched tensors
    size_t batch_size = 1;

    // We only support HEAAN backend for now
    auto backend = static_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(
        runtime::Backend::create("HE:HEAAN"));

    vector<float> x = read_binary_constant(
        file_util::path_join(HE_SERIALIZED_ZOO, "weights/x_test_4096.bin"), batch_size * 784);
    vector<float> y = read_binary_constant(
        file_util::path_join(HE_SERIALIZED_ZOO, "weights/y_test_4096.bin"), batch_size * 10);

    // Global stop watch
    stopwatch sw_global;
    sw_global.start();

    // Load graph
    stopwatch sw_load_model;
    sw_load_model.start();
    const string filename = "mnist_cryptonets_batch_" + to_string(batch_size);
    const string json_path = file_util::path_join(HE_SERIALIZED_ZOO, filename + ".json");
    const string json_string = file_util::read_file_to_string(json_path);
    shared_ptr<Function> f = deserialize(json_string);
    NGRAPH_INFO << "Deserialize graph";
    NGRAPH_INFO << "x size " << x.size();
    NGRAPH_INFO << "Inputs loaded";
    sw_load_model.stop();
    NGRAPH_INFO << "sw_load_model: " << sw_load_model.get_milliseconds() << "ms";

    // Create input tensorview and copy tensors; create output tensorviews
    stopwatch sw_encrypt_input;
    sw_encrypt_input.start();
    auto parameters = f->get_parameters();
    vector<shared_ptr<runtime::TensorView>> parameter_tvs;
    for (auto parameter : parameters)
    {
        auto& shape = parameter->get_shape();
        auto& type = parameter->get_element_type();
        auto parameter_cipher_tv = backend->create_tensor(type, shape);
        NGRAPH_INFO << "Creating input shape: " << join(shape, "x");

        if (shape == Shape{batch_size, 784})
        {
            NGRAPH_INFO << "Copying " << shape_size(shape) << " elements";
            NGRAPH_INFO << "x is " << x.size() << " elements";
            copy_data(parameter_cipher_tv, x);
            parameter_tvs.push_back(parameter_cipher_tv);
        }
        else
        {
            NGRAPH_INFO << "Invalid shape" << join(shape, "x");
            throw ngraph_error("Invalid shape " + shape_size(shape));
        }
    }

    auto results = f->get_results();
    vector<shared_ptr<runtime::TensorView>> result_tvs;
    for (auto result : results)
    {
        auto& shape = result->get_shape();
        auto& type = result->get_element_type();
        NGRAPH_INFO << "Creating output shape: " << join(shape, "x");
        result_tvs.push_back(backend->create_tensor(type, shape));
    }
    sw_encrypt_input.stop();
    NGRAPH_INFO << "sw_encrypt_input: " << sw_encrypt_input.get_milliseconds() << "ms";

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
    auto result = read_vector<float>(result_tvs[0]);
    sw_decrypt_output.stop();
    NGRAPH_INFO << "sw_decrypt_output: " << sw_decrypt_output.get_milliseconds() << "ms";

    // Stop global stop watch
    sw_global.stop();
    NGRAPH_INFO << "sw_global: " << sw_global.get_milliseconds() << "ms";

    // Print results
    NGRAPH_INFO << "[Summary]";
    NGRAPH_INFO << "sw_load_model: " << sw_load_model.get_milliseconds() << "ms";
    NGRAPH_INFO << "sw_encrypt_input: " << sw_encrypt_input.get_milliseconds() << "ms";
    NGRAPH_INFO << "sw_run_model: " << sw_run_model.get_milliseconds() << "ms";
    NGRAPH_INFO << "sw_decrypt_output: " << sw_decrypt_output.get_milliseconds() << "ms";
    NGRAPH_INFO << "sw_global: " << sw_global.get_milliseconds() << "ms";

    return 0;
}
