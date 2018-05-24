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

#include <vector>

#include "he_backend.hpp"
#include "he_heaan_backend.hpp"
#include "kernel/heaan/multiply_heaan.hpp"
#include "ngraph/type/element_type.hpp"
#include "heaan/heaan.hpp"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::heaan::multiply(const vector<shared_ptr<he::HeaanCiphertextWrapper>>& arg0,
                                   const vector<shared_ptr<he::HeaanCiphertextWrapper>>& arg1,
                                   vector<shared_ptr<he::HeaanCiphertextWrapper>>& out,
                                   const element::Type& type,
                                   shared_ptr<he_heaan::HEHeaanBackend> he_heaan_backend,
                                   size_t count)
{
    NGRAPH_INFO << "Vector cipher cipher heaan mult";
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        scalar_multiply(arg0[i], arg1[i], out[i], type, he_heaan_backend);
    }
}

void runtime::he::kernel::heaan::scalar_multiply(const shared_ptr<he::HeaanCiphertextWrapper>& arg0,
                                          const shared_ptr<he::HeaanCiphertextWrapper>& arg1,
                                          shared_ptr<he::HeaanCiphertextWrapper>& out,
                                          const element::Type& type,
                                          shared_ptr<he_heaan::HEHeaanBackend> he_heaan_backend)
{
    NGRAPH_INFO << "heaan scalar_multiply cipher cipher";
    out->m_ciphertext = he_heaan_backend->get_scheme()->mult(
        arg0->m_ciphertext, arg1->m_ciphertext);

}

void runtime::he::kernel::heaan::multiply(const vector<shared_ptr<he::HeaanCiphertextWrapper>>& arg0,
                                   const vector<shared_ptr<he::HeaanPlaintextWrapper>>& arg1,
                                   vector<shared_ptr<he::HeaanCiphertextWrapper>>& out,
                                   const element::Type& type,
                                   shared_ptr<he_heaan::HEHeaanBackend> he_heaan_backend,
                                   size_t count)
{
#pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        scalar_multiply(arg0[i], arg1[i], out[i], type, he_heaan_backend);
    }
}

void runtime::he::kernel::heaan::scalar_multiply(const shared_ptr<he::HeaanCiphertextWrapper>& arg0,
                                          const shared_ptr<he::HeaanPlaintextWrapper>& arg1,
                                          shared_ptr<he::HeaanCiphertextWrapper>& out,
                                          const element::Type& type,
                                          shared_ptr<he_heaan::HEHeaanBackend> he_heaan_backend)
{
    const string type_name = type.c_type_string();
    out->m_ciphertext = he_heaan_backend->get_scheme()->multByConst(
            arg0->m_ciphertext, arg1->m_plaintext, he_heaan_backend->get_precision());

    shared_ptr<runtime::he::HeaanPlaintextWrapper> x = make_shared<runtime::he::HeaanPlaintextWrapper>();
    he_heaan_backend->decrypt(x, arg0);
    auto tmp = dynamic_pointer_cast<he::HeaanPlaintextWrapper>(x);
    assert(tmp != nullptr);

    auto y = make_shared<runtime::he::HeaanPlaintextWrapper>();
    he_heaan_backend->decrypt(y, out);
    auto tmp2 = dynamic_pointer_cast<he::HeaanPlaintextWrapper>(y);
    assert(tmp2 != nullptr);

    NGRAPH_INFO << "Multiplying " << arg1->m_plaintext << " and " << tmp->m_plaintext;
    NGRAPH_INFO << "Result " << tmp2->m_plaintext;
}

void runtime::he::kernel::heaan::multiply(const vector<shared_ptr<he::HeaanPlaintextWrapper>>& arg0,
                                   const vector<shared_ptr<he::HeaanCiphertextWrapper>>& arg1,
                                   vector<shared_ptr<he::HeaanCiphertextWrapper>>& out,
                                   const element::Type& type,
                                   shared_ptr<he_heaan::HEHeaanBackend> he_heaan_backend,
                                   size_t count)
{
    multiply(arg1, arg0, out, type, he_heaan_backend, count);
}

void runtime::he::kernel::heaan::scalar_multiply(const shared_ptr<he::HeaanPlaintextWrapper>& arg0,
                                          const shared_ptr<he::HeaanCiphertextWrapper>& arg1,
                                          shared_ptr<he::HeaanCiphertextWrapper>& out,
                                          const element::Type& type,
                                          shared_ptr<he_heaan::HEHeaanBackend> he_heaan_backend)
{
    scalar_multiply(arg1, arg0, out, type, he_heaan_backend);
}

void runtime::he::kernel::heaan::multiply(const vector<shared_ptr<he::HeaanPlaintextWrapper>>& arg0,
                                   const vector<shared_ptr<he::HeaanPlaintextWrapper>>& arg1,
                                   vector<shared_ptr<he::HeaanPlaintextWrapper>>& out,
                                   const element::Type& type,
                                   shared_ptr<he_heaan::HEHeaanBackend> he_heaan_backend,
                                   size_t count)
{
    const string type_name = type.c_type_string();
    if (type_name != "float")
    {
        throw ngraph_error("Type " + type_name + " not supported");
    }

#pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
    {
        float x, y;
        he_heaan_backend->decode(&x, arg0[i], type);
        he_heaan_backend->decode(&y, arg1[i], type);
        float r = x * y;
        shared_ptr<he::HEPlaintext> out_he = dynamic_pointer_cast<he::HEPlaintext>(out[i]);
        he_heaan_backend->encode(out_he, &r, type);
        out[i] = dynamic_pointer_cast<he::HeaanPlaintextWrapper>(out_he);
    }
}

void runtime::he::kernel::heaan::scalar_multiply(const shared_ptr<he::HeaanPlaintextWrapper>& arg0,
                                          const shared_ptr<he::HeaanPlaintextWrapper>& arg1,
                                          shared_ptr<he::HeaanPlaintextWrapper>& out,
                                          const element::Type& type,
                                          shared_ptr<he_heaan::HEHeaanBackend> he_heaan_backend)
{
    const string type_name = type.c_type_string();
    if (type_name != "float")
    {
        throw ngraph_error("Type " + type_name + " not supported");
    }

    float x, y;
    he_heaan_backend->decode(&x, arg0, type);
    he_heaan_backend->decode(&y, arg1, type);
    float r = x * y;
    shared_ptr<he::HEPlaintext> out_he = dynamic_pointer_cast<he::HEPlaintext>(out);
    he_heaan_backend->encode(out_he, &r, type);
    out = dynamic_pointer_cast<he::HeaanPlaintextWrapper>(out_he);
}
