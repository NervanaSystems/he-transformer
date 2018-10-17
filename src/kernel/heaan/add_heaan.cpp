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

#include "kernel/heaan/add_heaan.hpp"
#include "he_heaan_backend.hpp"
#include "heaan_ciphertext_wrapper.hpp"
#include "heaan_plaintext_wrapper.hpp"
#include <fstream>

using namespace std;
using namespace ngraph;

void runtime::he::kernel::heaan::scalar_add(
    const shared_ptr<runtime::he::HeaanCiphertextWrapper>& arg0,
    const shared_ptr<runtime::he::HeaanCiphertextWrapper>& arg1,
    shared_ptr<runtime::he::HeaanCiphertextWrapper>& out,
    const element::Type& type,
    const shared_ptr<runtime::he::he_heaan::HEHeaanBackend> he_heaan_backend)
{
    /*if (out == arg0) // TODO: Discover why this is needed? (dot.cpp needs this)
    {
        out->m_ciphertext = he_heaan_backend->get_scheme()->add(arg1->m_ciphertext, arg0->m_ciphertext);
        //he_heaan_backend->get_scheme()->addAndEqual(out->m_ciphertext, arg1->m_ciphertext);
    }
    else
    { */
    /*if (arg0 == arg1)
    {
        NGRAPH_INFO << " (arg0 == arg1)";
    }
    if (arg0 == out)
    {
        NGRAPH_INFO << " (arg0 == out)";
    }
    if (arg1 == out)
    {
        NGRAPH_INFO << "arg1 == out !";
    }
     */
    out->m_ciphertext = he_heaan_backend->get_scheme()->add(arg1->m_ciphertext, arg0->m_ciphertext);

   auto plain = he_heaan_backend->create_empty_plaintext();
   he_heaan_backend->decrypt(plain, out);
   float plain_val = dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(plain)->m_plaintexts[0];

   if (plain_val > 1e50 || plain_val < -1e50)
   {
       #pragma omp critical
       {
            NGRAPH_INFO << "infail -> " << plain_val;
            print_ciphertext(arg0, he_heaan_backend, "./arg0_" + to_string(plain_val));
            print_ciphertext(arg1, he_heaan_backend, "./arg1_" + to_string(plain_val));
            print_ciphertext(out, he_heaan_backend, "./out_" + to_string(plain_val));

            auto secretKey = he_heaan_backend->get_secret_key();
            std::fstream fs;
            fs.open("secret_key_cipher.txt", std::fstream::in);
            fs << secretKey->sx << "\n";
            fs.close();

            exit(0);
       }

   }
}

void runtime::he::kernel::heaan::print_ciphertext(const shared_ptr<runtime::he::HeaanCiphertextWrapper>& out,
    const shared_ptr<runtime::he::he_heaan::HEHeaanBackend> he_heaan_backend,
    const std::string& name)
{
    std::fstream fs;
    fs.open(name + "_cipher.txt", std::fstream::in | std::fstream::out  | std::fstream::app);
    fs << name << " ciphertext" << "\n";
    fs << name << "logp " <<  out->m_ciphertext.logp << "\n";
    fs << name << "logq " <<  out->m_ciphertext.logp << "\n";
    fs << name << "slots " <<  out->m_ciphertext.slots << "\n";
    fs << name << "isComplex " <<  out->m_ciphertext.isComplex << "\n";
    auto plain = he_heaan_backend->create_empty_plaintext();
    he_heaan_backend->decrypt(plain, out);
    float plain_val = dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(plain)->m_plaintexts[0];
    fs << name << " plain value " << plain_val << "\n";
    fs << name << "ax " <<  out->m_ciphertext.ax << "\n";
    fs << name << "bx " <<  out->m_ciphertext.bx << "\n";
    fs.close();


}

void runtime::he::kernel::heaan::scalar_add(
    const shared_ptr<runtime::he::HeaanPlaintextWrapper>& arg0,
    const shared_ptr<runtime::he::HeaanPlaintextWrapper>& arg1,
    shared_ptr<runtime::he::HeaanPlaintextWrapper>& out,
    const element::Type& type,
    const shared_ptr<runtime::he::he_heaan::HEHeaanBackend> he_heaan_backend)
{
    const string type_name = type.c_type_string();
    if (type_name == "float")
    {
        float x, y;
        he_heaan_backend->decode(&x, arg0, type);
        he_heaan_backend->decode(&y, arg1, type);
        float r = x + y;
        shared_ptr<runtime::he::HEPlaintext> out_he =
            dynamic_pointer_cast<runtime::he::HEPlaintext>(out);
        he_heaan_backend->encode(out_he, &r, type);
        out = dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(out_he);
    }
    else if (type_name == "int64_t")
    {
        int64_t x, y;
        he_heaan_backend->decode(&x, arg0, type);
        he_heaan_backend->decode(&y, arg1, type);
        int64_t r = x + y;
        shared_ptr<runtime::he::HEPlaintext> out_he =
            dynamic_pointer_cast<runtime::he::HEPlaintext>(out);
        he_heaan_backend->encode(out_he, &r, type);
        out = dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(out_he);
    }
    else
    {
        throw ngraph_error("Type " + type_name + " not supported in HEAAN add.");
    }
}

void runtime::he::kernel::heaan::scalar_add(
    const shared_ptr<runtime::he::HeaanCiphertextWrapper>& arg0,
    const shared_ptr<runtime::he::HeaanPlaintextWrapper>& arg1,
    shared_ptr<runtime::he::HeaanCiphertextWrapper>& out,
    const element::Type& type,
    const shared_ptr<runtime::he::he_heaan::HEHeaanBackend> he_heaan_backend)
{
    out->m_ciphertext =
        he_heaan_backend->get_scheme()->addConst(arg0->m_ciphertext, arg1->m_plaintexts[0]);
}

void runtime::he::kernel::heaan::scalar_add(
    const shared_ptr<runtime::he::HeaanPlaintextWrapper>& arg0,
    const shared_ptr<runtime::he::HeaanCiphertextWrapper>& arg1,
    shared_ptr<runtime::he::HeaanCiphertextWrapper>& out,
    const element::Type& type,
    const shared_ptr<runtime::he::he_heaan::HEHeaanBackend> he_heaan_backend)
{
    scalar_add(arg1, arg0, out, type, he_heaan_backend);
}
