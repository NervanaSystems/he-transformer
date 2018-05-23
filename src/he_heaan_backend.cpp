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
    throw ngraph_error("HEHeaanBackend::create_valued_ciphertext unimplemented");
    /* const string type_name = element_type.c_type_string();
    auto ciphertext =
        dynamic_pointer_cast<runtime::he::HeaanCiphertextWrapper>(create_empty_ciphertext());
    if (ciphertext == nullptr)
    {
        throw ngraph_error("Ciphertext is not heaan ciphertext in create_valued_ciphertext");
    }
    if (type_name == "float")
    {
        heaan::Plaintext plaintext = m_frac_encoder->encode(value);
        m_encryptor->encrypt(plaintext, ciphertext->m_ciphertext);
    }
    else if (type_name == "int64_t")
    {
        heaan::Plaintext plaintext = m_int_encoder->encode(static_cast<int64_t>(value));
        m_encryptor->encrypt(plaintext, ciphertext->m_ciphertext);
    }
    else
    {
        throw ngraph_error("Type not supported at create_ciphertext");
    }
    return ciphertext; */
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
        //output =
        //    make_shared<runtime::he::HeaanPlaintextWrapper>(m_scheme->encodeSingle(*(double*)input, m_logp, m_logq));
    }
    else if (type_name == "int64_t")
    {
        NGRAPH_INFO << "Encoding " << (double)*(int64_t*)input;
        //output =
        //    make_shared<runtime::he::HeaanPlaintextWrapper>(m_scheme->encodeSingle((double)*(int64_t*)input, m_logp, m_logq));

        NGRAPH_INFO << "m_logq " << m_context->logQ << ", m_logp " << m_log_precision << ", logN " << m_context->logN;
        auto tmp1 = m_scheme->encryptSingle(5.1, m_log_precision, m_context->logQ);
        auto tmp2 = m_scheme->decryptSingle(*m_secret_key, tmp1);
        NGRAPH_INFO << "decodes to " << tmp2.real() << " , " << tmp2.imag() ;

        auto tmp3 = m_scheme->encodeSingle(5.0, m_log_precision, m_context->logQ);
        NGRAPH_INFO << "Encoded 5.0 to " << tmp3.mx << " isComplex? " << tmp3.isComplex << ", logp " << tmp3.logp
                    << ", tmp3.logq " << tmp3.logq;
        auto tmp4 = m_scheme->decodeSingle(tmp3);
        NGRAPH_INFO << "decodes to " << tmp4.real() << " , " << tmp4.imag() ;

        long logp = m_log_precision;
        auto rr = to_RR(5.0);
        NGRAPH_INFO << "rr.x " << rr.x << " rr.e " << rr.e;
        RR rrr = MakeRR(rr.x, rr.e + logp);
        NGRAPH_INFO << "rr.x " << rrr.x << " rr.e " << rrr.e << " rrr " << rrr;

        throw ngraph_error("tmp");

        long logq = m_context->logQ;
        bool isComplex = false;
        auto q = m_context->qpowvec[logq];
        auto mx = tmp3.mx;
        cout << "q " << q << endl;
        cout << "mx.rep[0] " << mx.rep[0] << endl;
        auto tmp = mx.rep[0] % q;
        cout << "NumBit == ? " << (NumBits(tmp) == logq) << "?" << endl;
        complex<double> res;
        NGRAPH_INFO << "tmp " << tmp;
        RR xp = to_RR(tmp);
        cout << "xp " << xp << endl;
        cout << "xp " << xp << endl;
        xp.e -= logp;
        cout << "xp " << xp << endl;
        //res.real(heaan::EvaluatorUtils::scaleDownToReal(tmp, logp));
        //cout << "res.real() " << res.real() << endl;

        throw ngraph_error("tmp");

        /* auto plain = dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(output);
        assert(plain != nullptr);
        auto tmp = m_scheme->decodeSingle(plain->m_plaintext);
        NGRAPH_INFO << "decodes to " << tmp.real() << " , " << tmp.imag() ; */
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
            auto tmp = m_scheme->decodeSingle(heaan_input->m_plaintext);
            NGRAPH_INFO << "Decoding " << tmp.real() << " , " << tmp.imag();
            int64_t x = (int64_t)(m_scheme->decodeSingle(heaan_input->m_plaintext)).real();
            memcpy(output, &x, type.size());
        }
        else if (type_name == "float")
        {
            float x = (float)(m_scheme->decodeSingle(heaan_input->m_plaintext)).real();
            memcpy(output, &x, type.size());
        }
        else if (type_name == "double")
        {
            double x = (double)(m_scheme->decodeSingle(heaan_input->m_plaintext)).real();
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
        NGRAPH_INFO << "Encrypting";
        heaan_output->m_ciphertext = m_scheme->encryptMsg(heaan_input->m_plaintext);
        NGRAPH_INFO << "Done Encrypting";
    }
    else
    {
        throw ngraph_error("HEHeaanBackend::encrypt has non-heaan ciphertexts");
    }
}

void runtime::he::he_heaan::HEHeaanBackend::decrypt(shared_ptr<runtime::he::HEPlaintext> output,
                                                    const shared_ptr<he::HECiphertext> input)
{
    throw ngraph_error("HEHeaanBackend::decrypt not implemented");
    /* auto heaan_output = dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(output);
    auto heaan_input = dynamic_pointer_cast<runtime::he::HeaanCiphertextWrapper>(input);
    m_decryptor->decrypt(heaan_input->m_ciphertext, heaan_output->m_plaintext); */
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
