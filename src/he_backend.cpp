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

#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/function.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/manager.hpp"

#include "he_backend.hpp"
#include "he_call_frame.hpp"
#include "he_cipher_tensor_view.hpp"
#include "he_plain_tensor_view.hpp"
#include "he_tensor_view.hpp"
#include "pass/insert_relinearize.hpp"

using namespace ngraph;
using namespace std;

static void print_seal_context(const seal::SEALContext& context)
{
    NGRAPH_INFO << endl
                << "/ Encryption parameters:" << endl
                << "| poly_modulus: " << context.poly_modulus().to_string() << endl
                // Print the size of the true (product) coefficient modulus
                << "| coeff_modulus size: " << context.total_coeff_modulus().significant_bit_count()
                << " bits" << endl
                << "| plain_modulus: " << context.plain_modulus().value() << endl
                << "\\ noise_standard_deviation: " << context.noise_standard_deviation();
}

runtime::he::HEBackend::HEBackend()
    : runtime::he::HEBackend(runtime::he::default_seal_parameter)
{
}

runtime::he::HEBackend::HEBackend(const runtime::he::SEALParameter& sp)
{
    assert_valid_seal_parameter(sp);

    // Context
    m_context = make_seal_context(sp);
    print_seal_context(*m_context);

    // Encoders
    m_int_encoder = make_shared<seal::IntegerEncoder>(m_context->plain_modulus());
    m_frac_encoder =
        make_shared<seal::FractionalEncoder>(m_context->plain_modulus(),
                                             m_context->poly_modulus(),
                                             sp.fractional_encoder_integer_coeff_count,
                                             sp.fractional_encoder_fraction_coeff_count,
                                             sp.fractional_encoder_base);

    // Keygen, encryptor and decryptor
    m_keygen = make_shared<seal::KeyGenerator>(*m_context);
    m_public_key = make_shared<seal::PublicKey>(m_keygen->public_key());
    m_secret_key = make_shared<seal::SecretKey>(m_keygen->secret_key());
    m_encryptor = make_shared<seal::Encryptor>(*m_context, *m_public_key);
    m_decryptor = make_shared<seal::Decryptor>(*m_context, *m_secret_key);

    // Evaluator
    seal::EvaluationKeys ev_key;
    m_keygen->generate_evaluation_keys(sp.evaluation_decomposition_bit_count, ev_key);
    m_ev_key = make_shared<seal::EvaluationKeys>(ev_key);
    m_evaluator = make_shared<seal::Evaluator>(*m_context);

    // Plaintext constants
    m_plaintext_num = plaintext_num{m_frac_encoder->encode(1),
                                    m_frac_encoder->encode(-1),
                                    m_int_encoder->encode(1),
                                    m_int_encoder->encode(-1)};
}

runtime::he::HEBackend::~HEBackend()
{
}

shared_ptr<runtime::TensorView>
    runtime::he::HEBackend::create_tensor(const element::Type& element_type, const Shape& shape)
{
    shared_ptr<HEBackend> he_backend = shared_from_this();
    auto rc = make_shared<runtime::he::HECipherTensorView>(element_type, shape, he_backend);
    return static_pointer_cast<runtime::TensorView>(rc);
}

shared_ptr<runtime::TensorView> runtime::he::HEBackend::create_constant_tensor(
    const element::Type& element_type, const Shape& shape, size_t element)
{
    shared_ptr<runtime::TensorView> tensor = create_tensor(element_type, shape);
    shared_ptr<runtime::he::HECipherTensorView> cipher_tensor =
        static_pointer_cast<runtime::he::HECipherTensorView>(tensor);

    size_t num_elements = shape_size(shape);
    size_t bytes_to_write = num_elements * element_type.size();

    const string type_name = element_type.c_type_string();

    if (type_name == "float")
    {
        vector<float> elements;
        for (size_t i = 0; i < num_elements; ++i)
        {
            elements.push_back(element);
        }
        cipher_tensor->write((void*)&elements[0], 0, bytes_to_write);
    }
    else if (type_name == "int64_t")
    {
        vector<int64_t> elements;
        for (size_t i = 0; i < num_elements; ++i)
        {
            elements.push_back(element);
        }
        cipher_tensor->write((void*)&elements[0], 0, bytes_to_write);
    }
    else if (type_name == "uint64_t")
    {
        vector<uint64_t> elements;
        for (size_t i = 0; i < num_elements; ++i)
        {
            elements.push_back(element);
        }
        cipher_tensor->write((void*)&elements[0], 0, bytes_to_write);
    }
    else
    {
        throw ngraph_error("Type not supported at create_constant_tensor");
    }

    return static_pointer_cast<runtime::TensorView>(cipher_tensor);
}

shared_ptr<runtime::TensorView>
    runtime::he::HEBackend::create_zero_tensor(const element::Type& element_type,
                                               const Shape& shape)
{
    return create_constant_tensor(element_type, shape, 0);
}

shared_ptr<runtime::TensorView>
    runtime::he::HEBackend::create_ones_tensor(const element::Type& element_type,
                                               const Shape& shape)
{
    return create_constant_tensor(element_type, shape, 1);
}
/* {
    shared_ptr<runtime::TensorView> tensor = create_tensor(element_type, shape);
    shared_ptr<runtime::he::HECipherTensorView> cipher_tensor =
        static_pointer_cast<runtime::he::HECipherTensorView>(tensor);

    size_t num_elements = shape_size(shape);
    size_t bytes_to_write = num_elements * element_type.size();

    const string type_name = element_type.c_type_string();

    if (type_name == "float")
    {
        vector<float> zero;
        for (size_t i = 0; i < num_elements; ++i)
        {
            zero.push_back(0);
        }
        cipher_tensor->write((void*)&zero[0], 0, bytes_to_write);
    }
    else if (type_name == "int64_t")
    {
        vector<int64_t> zero;
        for (size_t i = 0; i < num_elements; ++i)
        {
            zero.push_back(0);
        }
        cipher_tensor->write((void*)&zero[0], 0, bytes_to_write);
    }
    else if (type_name == "uint64_t")
    {
        vector<uint64_t> zero;
        for (size_t i = 0; i < num_elements; ++i)
        {
            zero.push_back(0);
        }
        cipher_tensor->write((void*)&zero[0], 0, bytes_to_write);
    }
    else
    {
        throw ngraph_error("Type not supported at create_zero_tensor");
    }

    return static_pointer_cast<runtime::TensorView>(cipher_tensor);
} */

shared_ptr<runtime::TensorView>
    runtime::he::HEBackend::create_plain_tensor(const element::Type& element_type,
                                                const Shape& shape)
{
    shared_ptr<HEBackend> he_backend = shared_from_this();
    auto rc = make_shared<runtime::he::HEPlainTensorView>(element_type, shape, he_backend);
    return static_pointer_cast<runtime::TensorView>(rc);
}

shared_ptr<runtime::TensorView> runtime::he::HEBackend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    throw ngraph_error("HE create_tensor unimplemented");
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
    auto call_frame = m_function_map.at(func);
    call_frame->call(outputs, inputs);
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

void runtime::he::HEBackend::encode(seal::Plaintext& output,
                                    const void* input,
                                    const element::Type& type)
{
    const string type_name = type.c_type_string();

    if (type_name == "int64_t")
    {
        output = m_int_encoder->encode(*(int64_t*)input);
    }
    else if (type_name == "uint64_t")
    {
        output = m_int_encoder->encode(*(uint64_t*)input);
    }
    else if (type_name == "float")
    {
        output = m_frac_encoder->encode(*(float*)input);
    }
    else
    {
        NGRAPH_INFO << "Unsupported element type in decode " << type_name;
        throw ngraph_error("Unsupported element type " + type_name);
    }
}

void runtime::he::HEBackend::decode(void* output,
                                    const seal::Plaintext& input,
                                    const element::Type& type)
{
    const string type_name = type.c_type_string();

    if (type_name == "int64_t")
    {
        int64_t x = m_int_encoder->decode_int64(input);
        memcpy(output, &x, type.size());
    }
    else if (type_name == "uint64_t")
    {
        uint64_t x = m_int_encoder->decode_int64(input);
        memcpy(output, &x, type.size());
    }
    else if (type_name == "float")
    {
        float x = m_frac_encoder->decode(input);
        memcpy(output, &x, type.size());
    }
    else
    {
        NGRAPH_INFO << "Unsupported element type in decode " << type_name;
        throw ngraph_error("Unsupported element type " + type_name);
    }
}

void runtime::he::HEBackend::encrypt(seal::Ciphertext& output, const seal::Plaintext& input)
{
    m_encryptor->encrypt(input, output);
}

void runtime::he::HEBackend::decrypt(seal::Plaintext& output, const seal::Ciphertext& input)
{
    m_decryptor->decrypt(input, output);
}

int runtime::he::HEBackend::noise_budget(const shared_ptr<seal::Ciphertext>& ciphertext)
{
    return m_decryptor->invariant_noise_budget(*ciphertext);
}

void runtime::he::HEBackend::check_noise_budget(
    const vector<shared_ptr<runtime::he::HETensorView>>& tvs)
{
    // Check noise budget
    NGRAPH_INFO << "Checking noise budget ";
#pragma omp parallel for
    for (size_t i = 0; i < tvs.size(); ++i)
    {
        shared_ptr<HECipherTensorView> out_i = dynamic_pointer_cast<HECipherTensorView>(tvs[i]);
        if (out_i != nullptr)
        {
            for (shared_ptr<seal::Ciphertext> ciphertext : out_i->get_elements())
            {
                int budget = noise_budget(ciphertext);
                NGRAPH_INFO << "Noise budget " << budget;
                if (budget <= 0)
                {
                    throw ngraph_error("Noise budget depleted");
                }
                break; // TODO: remove
            }
        }
    }
    NGRAPH_INFO << "Done checking noise budget ";
}
