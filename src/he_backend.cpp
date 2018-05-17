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

#include <limits>

#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/function.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"

#include "he_backend.hpp"
#include "he_call_frame.hpp"
#include "he_cipher_tensor_view.hpp"
#include "he_plain_tensor_view.hpp"
#include "he_tensor_view.hpp"
#include "pass/insert_relinearize.hpp"

using namespace ngraph;
using namespace std;

runtime::he::HEBackend::HEBackend()
{
    throw ngraph_error("HEBackend not implemented");
}

runtime::he::HEBackend::HEBackend(const runtime::he::HEParameter& hep)
{
    throw ngraph_error("HEBackend() not implemented");
}

shared_ptr<runtime::TensorView>
    runtime::he::HEBackend::create_tensor(const element::Type& element_type, const Shape& shape)
{
    throw ngraph_error("HE create_tensor unimplemented");
}

shared_ptr<runtime::TensorView> runtime::he::HEBackend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    throw ngraph_error("HE create_tensor unimplemented");
}

shared_ptr<runtime::TensorView>
    runtime::he::HEBackend::create_plain_tensor(const element::Type& element_type,
                                                const Shape& shape)
{
    throw ngraph_error("HE create_plain_tensor unimplemented");
}

std::shared_ptr<he::Ciphertext> runtime::he::HEBackend::create_valued_ciphertext(
    float value, const element::Type& element_type, const he::MemoryPoolHandle& pool) const
{
    throw ngraph_error("HE create_valued_ciphertextunimplemented");
}

std::shared_ptr<he::Ciphertext>
    runtime::he::HEBackend::create_empty_ciphertext(const he::MemoryPoolHandle& pool) const
{
    throw ngraph_error("HE create_empty_ciphertext unimplemented");
}

std::shared_ptr<he::Plaintext> runtime::he::HEBackend::create_valued_plaintext(
    float value, const element::Type& element_type, const he::MemoryPoolHandle& pool) const
{
    throw ngraph_error("HE create_valued_plaintext unimplemented");
}

std::shared_ptr<he::Plaintext>
    runtime::he::HEBackend::create_empty_plaintext(const he::MemoryPoolHandle& pool) const
{
    throw ngraph_error("HE create_empty_plaintext unimplemented");
}

std::shared_ptr<he::Ciphertext>
    runtime::he::HEBackend::create_valued_ciphertext(float value,
                                                     const element::Type& element_type) const
{
    throw ngraph_error("HE create_empty_ciphertext unimplemented");
}

std::shared_ptr<he::Ciphertext> runtime::he::HEBackend::create_empty_ciphertext() const
{
    throw ngraph_error("HE create_empty_ciphertext unimplemented");
}

std::shared_ptr<he::Plaintext>
    runtime::he::HEBackend::create_valued_plaintext(float value,
                                                    const element::Type& element_type) const
{
    throw ngraph_error("HE create_valued_plaintext unimplemented");
}

std::shared_ptr<he::Plaintext> runtime::he::HEBackend::create_empty_plaintext() const
{
    throw ngraph_error("HE create_empty_plaintext unimplemented");
}

shared_ptr<runtime::TensorView> runtime::he::HEBackend::create_valued_tensor(
    float value, const element::Type& element_type, const Shape& shape)
{
    throw ngraph_error("HE create_valued_tensor unimplemented");
}

shared_ptr<runtime::TensorView> runtime::he::HEBackend::create_valued_plain_tensor(
    float value, const element::Type& element_type, const Shape& shape)
{
    throw ngraph_error("HE create_valued_plain_tensor unimplemented");
}

bool runtime::he::HEBackend::compile(shared_ptr<Function> func)
{
    if (m_function_map.count(func) == 0)
    {
        shared_ptr<HEBackend> he_backend = shared_from_this();
        shared_ptr<Function> cf_func = clone_function(*func);

        // Run passes
        ngraph::pass::Manager pass_manager;
        pass_manager
            .register_pass<ngraph::pass::AssignLayout<descriptor::layout::DenseTensorViewLayout>>();
        pass_manager.register_pass<runtime::he::pass::InsertRelinearize>();
        pass_manager.run_passes(cf_func);

        // Create call frame
        shared_ptr<HECallFrame> call_frame = make_shared<HECallFrame>(cf_func, he_backend);

        m_function_map.insert({func, call_frame});
    }
    return true;
}

bool runtime::he::HEBackend::call(shared_ptr<Function> func,
                                  const vector<shared_ptr<runtime::TensorView>>& outputs,
                                  const vector<shared_ptr<runtime::TensorView>>& inputs)
{
    compile(func);
    m_function_map.at(func)->call(outputs, inputs);
    return true;
}

void runtime::he::HEBackend::clear_function_instance()
{
    m_function_map.clear();
}

void runtime::he::HEBackend::remove_compiled_function(shared_ptr<Function> func)
{
    throw ngraph_error("HEBackend remove compile function unimplemented");
}

void runtime::he::HEBackend::encode(he::Plaintext& output,
                                    const void* input,
                                    const element::Type& type)
{
    throw ngraph_error("HEBackend encode unimplemented");
}

void runtime::he::HEBackend::decode(void* output,
                                    const he::Plaintext& input,
                                    const element::Type& type)
{
    throw ngraph_error("HEBackend decode unimplemented");

}

void runtime::he::HEBackend::encrypt(he::Ciphertext& output, const he::Plaintext& input)
{
    throw ngraph_error("HEBackend encrypt unimplemented");
}

void runtime::he::HEBackend::decrypt(he::Plaintext& output, const he::Ciphertext& input)
{
    throw ngraph_error("HEBackend decrypt unimplemented");
}


void runtime::he::HEBackend::enable_performance_data(shared_ptr<Function> func, bool enable)
{
    // Enabled by default
}

vector<runtime::PerformanceCounter>
    runtime::he::HEBackend::get_performance_data(shared_ptr<Function> func) const
{
    return m_function_map.at(func)->get_performance_data();
}

void runtime::he::HEBackend::visualize_function_after_pass(const shared_ptr<Function>& func,
                                                           const string& file_name)
{
    compile(func);
    auto cf = m_function_map.at(func);
    auto compiled_func = cf->get_compiled_function();
    NGRAPH_INFO << "Visualize graph to " << file_name;
    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<ngraph::pass::VisualizeTree>(file_name);
    pass_manager.run_passes(compiled_func);
}
