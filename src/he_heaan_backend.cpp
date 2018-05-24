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
#include <math.h>

#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/function.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"

#include "he_backend.hpp"
#include "he_call_frame.hpp"
#include "he_cipher_tensor_view.hpp"
#include "he_heaan_backend.hpp"
#include "he_heaan_parameter.hpp"
#include "he_plain_tensor_view.hpp"
#include "he_tensor_view.hpp"
#include "heaan_ciphertext_wrapper.hpp"
#include "heaan_plaintext_wrapper.hpp"
#include "pass/insert_relinearize.hpp"

using namespace ngraph;
using namespace std;

static void print_heaan_context(const heaan::Context& context)
{
    NGRAPH_INFO << endl
                << "/ Encryption parameters:" << endl
                << "| poly_modulus: " << context.N << endl
                << "| plain_modulus: " << context.Q << endl
                << "\\ noise_standard_deviation: " << context.sigma;
}

runtime::he::he_heaan::HEHeaanBackend::HEHeaanBackend()
    : runtime::he::he_heaan::HEHeaanBackend(
          make_shared<runtime::he::HEHeaanParameter>(runtime::he::default_heaan_parameter))
{
}

runtime::he::he_heaan::HEHeaanBackend::HEHeaanBackend(
    const shared_ptr<runtime::he::HEParameter> hep)
    : runtime::he::he_heaan::HEHeaanBackend(
          make_shared<runtime::he::HEHeaanParameter>(hep->m_poly_modulus, hep->m_plain_modulus))
{
}

runtime::he::he_heaan::HEHeaanBackend::HEHeaanBackend(
    const shared_ptr<runtime::he::HEHeaanParameter> sp)
{
    // Context
    m_context = make_shared<heaan::Context>(sp->m_poly_modulus, sp->m_plain_modulus);
    print_heaan_context(*m_context);

    m_log_precision = (long)sp->m_log_precision;

    // Secret Key
    m_secret_key = make_shared<heaan::SecretKey>(m_context->logN);

    // Scheme
    m_scheme = make_shared<heaan::Scheme>(*m_secret_key, *m_context);


    // Keygen, encryptor and decryptor
    // Evaluator
    // Plaintext constants

    NGRAPH_INFO << "Created Heaan backend";
}

runtime::he::he_heaan::HEHeaanBackend::~HEHeaanBackend()
{
}

shared_ptr<runtime::TensorView>
    runtime::he::he_heaan::HEHeaanBackend::create_tensor(const element::Type& element_type,
                                                         const Shape& shape)
{
    shared_ptr<HEHeaanBackend> he_heaan_backend =
        dynamic_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(shared_from_this());
    auto rc = make_shared<runtime::he::HECipherTensorView>(element_type, shape, he_heaan_backend);
    return static_pointer_cast<runtime::TensorView>(rc);
}

shared_ptr<runtime::TensorView> runtime::he::he_heaan::HEHeaanBackend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    throw ngraph_error("HEHeaan create_tensor unimplemented");
}

shared_ptr<runtime::TensorView>
    runtime::he::he_heaan::HEHeaanBackend::create_plain_tensor(const element::Type& element_type,
                                                               const Shape& shape)
{
    shared_ptr<HEHeaanBackend> he_heaan_backend =
        dynamic_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(shared_from_this());
    auto rc = make_shared<runtime::he::HEPlainTensorView>(element_type, shape, he_heaan_backend);
    return static_pointer_cast<runtime::TensorView>(rc);
}

shared_ptr<runtime::he::HECiphertext>
    runtime::he::he_heaan::HEHeaanBackend::create_valued_ciphertext(
        float value, const element::Type& element_type) const
{
    auto ciphertext =
        dynamic_pointer_cast<runtime::he::HeaanCiphertextWrapper>(create_empty_ciphertext());

    ciphertext->m_ciphertext = m_scheme->encryptSingle((double)value, get_precision(), m_context->logQ);

    return ciphertext;
}

shared_ptr<runtime::he::HECiphertext>
    runtime::he::he_heaan::HEHeaanBackend::create_empty_ciphertext() const
{
    return make_shared<runtime::he::HeaanCiphertextWrapper>();
}

shared_ptr<runtime::he::HEPlaintext> runtime::he::he_heaan::HEHeaanBackend::create_valued_plaintext(
    float value, const element::Type& element_type) const
{
    throw ngraph_error("HEHeaanBackend::create_valued_plaintext unimplemented");
    /* const string type_name = element_type.c_type_string();
    shared_ptr<runtime::he::HEPlaintext> plaintext = create_empty_plaintext();
    auto plaintext_heaan = dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(plaintext);
    if (plaintext_heaan == nullptr)
    {
        NGRAPH_INFO << "plaintext is not heaan type in create_valued_plaintext";
    }
    if (type_name == "float")
    {
        plaintext_heaan->m_plaintext = m_frac_encoder->encode(value);
    }
    else if (type_name == "int64_t")
    {
        plaintext_heaan->m_plaintext = m_int_encoder->encode(static_cast<int64_t>(value));
    }
    else
    {
        throw ngraph_error("Type not supported at create_ciphertext");
    }
    return plaintext; */
}

shared_ptr<runtime::he::HEPlaintext>
    runtime::he::he_heaan::HEHeaanBackend::create_empty_plaintext() const
{
    return make_shared<HeaanPlaintextWrapper>();
}

shared_ptr<runtime::TensorView> runtime::he::he_heaan::HEHeaanBackend::create_valued_tensor(
    float value, const element::Type& element_type, const Shape& shape)
{
    auto tensor = static_pointer_cast<HECipherTensorView>(create_tensor(element_type, shape));
    vector<shared_ptr<runtime::he::HECiphertext>>& cipher_texts = tensor->get_elements();
#pragma omp parallel for
    for (size_t i = 0; i < cipher_texts.size(); ++i)
    {
        cipher_texts[i] = create_valued_ciphertext(value, element_type);
    }
    return tensor;
}

shared_ptr<runtime::TensorView> runtime::he::he_heaan::HEHeaanBackend::create_valued_plain_tensor(
    float value, const element::Type& element_type, const Shape& shape)
{
    throw ngraph_error("create_valued_tensor plain unimplemented in heaan");
    /* auto tensor = static_pointer_cast<HEPlainTensorView>(create_plain_tensor(element_type, shape));
    vector<shared_ptr<heaan::Plaintext>>& plain_texts = tensor->get_elements();
 #pragma omp parallel for
    for (size_t i = 0; i < plain_texts.size(); ++i)
    {
        heaan::MemoryPoolHandle pool = heaan::MemoryPoolHandle::New(false);
        plain_texts[i] = create_valued_plaintext(value, element_type, pool);
    }
    return tensor; */
}

bool runtime::he::he_heaan::HEHeaanBackend::compile(shared_ptr<Function> func)
{
    if (m_function_map.count(func) == 0)
    {
        shared_ptr<HEHeaanBackend> he_heaan_backend =
            dynamic_pointer_cast<runtime::he::he_heaan::HEHeaanBackend>(shared_from_this());
        shared_ptr<Function> cf_func = clone_function(*func);

        // Run passes
        ngraph::pass::Manager pass_manager;
        pass_manager
            .register_pass<ngraph::pass::AssignLayout<descriptor::layout::DenseTensorViewLayout>>();
        pass_manager.register_pass<runtime::he::pass::InsertRelinearize>();
        pass_manager.run_passes(cf_func);

        // Create call frame
        shared_ptr<HECallFrame> call_frame = make_shared<HECallFrame>(cf_func, he_heaan_backend);

        m_function_map.insert({func, call_frame});
    }
    return true;
}

bool runtime::he::he_heaan::HEHeaanBackend::call(
    shared_ptr<Function> func,
    const vector<shared_ptr<runtime::TensorView>>& outputs,
    const vector<shared_ptr<runtime::TensorView>>& inputs)
{
    compile(func);
    m_function_map.at(func)->call(outputs, inputs);
    return true;
}

void runtime::he::he_heaan::HEHeaanBackend::clear_function_instance()
{
    m_function_map.clear();
}

void runtime::he::he_heaan::HEHeaanBackend::remove_compiled_function(shared_ptr<Function> func)
{
    throw ngraph_error("HEHeaanBackend remove compile function unimplemented");
}

void runtime::he::he_heaan::HEHeaanBackend::encode(shared_ptr<runtime::he::HEPlaintext>& output,
                                                   const void* input,
                                                   const element::Type& type)
{
    const string type_name = type.c_type_string();

    if (type_name == "double")
    {
        output = make_shared<runtime::he::HeaanPlaintextWrapper>(*(double*)input);
    }
    else if (type_name == "int64_t")
    {
        output = make_shared<runtime::he::HeaanPlaintextWrapper>((double)*(int64_t*)input);
    }
    else if (type_name == "float")
    {
        output = make_shared<runtime::he::HeaanPlaintextWrapper>((double)*(float*)input);
    }
    else
    {
        NGRAPH_INFO << "Unsupported element type in encode " << type_name;
        throw ngraph_error("Unsupported element type " + type_name);
    }
}

void runtime::he::he_heaan::HEHeaanBackend::decode(void* output,
                                                   const shared_ptr<runtime::he::HEPlaintext> input,
                                                   const element::Type& type)
{
    const string type_name = type.c_type_string();

    if (auto heaan_input = dynamic_pointer_cast<HeaanPlaintextWrapper>(input))
    {
        if (type_name == "int64_t")
        {
            int64_t x = std::round(heaan_input->m_plaintext);
            memcpy(output, &x, type.size());
        }
        else if (type_name == "float")
        {
            float x = heaan_input->m_plaintext;
            memcpy(output, &x, type.size());
        }
        else if (type_name == "double")
        {
            double x = (double)heaan_input->m_plaintext;
            memcpy(output, &x, type.size());
        }
        else
        {
            NGRAPH_INFO << "Unsupported element type in decode " << type_name;
            throw ngraph_error("Unsupported element type " + type_name);
        }
    }
    else
    {
        throw ngraph_error("HEHeaanBackend::decode input is not heaan plaintext");
    }
}

void runtime::he::he_heaan::HEHeaanBackend::encrypt(
    shared_ptr<runtime::he::HECiphertext> output, const shared_ptr<runtime::he::HEPlaintext> input)
{
    auto heaan_output = dynamic_pointer_cast<runtime::he::HeaanCiphertextWrapper>(output);
    auto heaan_input = dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(input);
    if (heaan_output != nullptr && heaan_input != nullptr)
    {
        heaan_output->m_ciphertext = m_scheme->encryptSingle(heaan_input->m_plaintext, m_log_precision, m_context->logQ);
    }
    else
    {
        throw ngraph_error("HEHeaanBackend::encrypt has non-heaan ciphertexts");
    }
}

void runtime::he::he_heaan::HEHeaanBackend::decrypt(shared_ptr<runtime::he::HEPlaintext> output,
                                                    const shared_ptr<he::HECiphertext> input)
{
    auto heaan_output = dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(output);
    auto heaan_input = dynamic_pointer_cast<runtime::he::HeaanCiphertextWrapper>(input);
    if (heaan_output != nullptr && heaan_input != nullptr)
    {
        heaan_output->m_plaintext = m_scheme->decryptSingle(*m_secret_key, heaan_input->m_ciphertext).real();
    }
    else
    {
        throw ngraph_error("HEHeaanBackend::decrypt has non-heaan ciphertexts");
    }
}

void runtime::he::he_heaan::HEHeaanBackend::enable_performance_data(shared_ptr<Function> func,
                                                                    bool enable)
{
    // Enabled by default
}

vector<runtime::PerformanceCounter>
    runtime::he::he_heaan::HEHeaanBackend::get_performance_data(shared_ptr<Function> func) const
{
    return m_function_map.at(func)->get_performance_data();
}

void runtime::he::he_heaan::HEHeaanBackend::visualize_function_after_pass(
    const shared_ptr<Function>& func, const string& file_name)
{
    compile(func);
    auto cf = m_function_map.at(func);
    auto compiled_func = cf->get_compiled_function();
    NGRAPH_INFO << "Visualize graph to " << file_name;
    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<ngraph::pass::VisualizeTree>(file_name);
    pass_manager.run_passes(compiled_func);
}
