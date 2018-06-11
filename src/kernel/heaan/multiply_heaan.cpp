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

#include "kernel/heaan/multiply_heaan.hpp"
#include "he_heaan_backend.hpp"

using namespace std;
using namespace ngraph;

void runtime::he::kernel::heaan::scalar_multiply(
    const shared_ptr<runtime::he::HeaanCiphertextWrapper>& arg0,
    const shared_ptr<runtime::he::HeaanCiphertextWrapper>& arg1,
    shared_ptr<runtime::he::HeaanCiphertextWrapper>& out,
    const element::Type& type,
    const shared_ptr<runtime::he::he_heaan::HEHeaanBackend> he_heaan_backend)
{
    NGRAPH_INFO << "Scalar multiply cipher cipher";
    out->m_ciphertext =
        he_heaan_backend->get_scheme()->mult(arg0->m_ciphertext, arg1->m_ciphertext);
    //TODO: reScaleByAndEqual?
}

void runtime::he::kernel::heaan::scalar_multiply(
    const shared_ptr<runtime::he::HeaanCiphertextWrapper>& arg0,
    const shared_ptr<runtime::he::HeaanPlaintextWrapper>& arg1,
    shared_ptr<runtime::he::HeaanCiphertextWrapper>& out,
    const element::Type& type,
    const shared_ptr<runtime::he::he_heaan::HEHeaanBackend> he_heaan_backend)
{
    const string type_name = type.c_type_string();
    NGRAPH_INFO << "Scalar multiply cipher plain";
    NGRAPH_INFO << "arg1->m_plaintexts.size() " << arg1->m_plaintexts.size();
    vector<double> tmp{arg1->m_plaintexts[0], arg1->m_plaintexts[0]};
    NGRAPH_INFO << "Plaintext: " << tmp[0] << " " << tmp[1];

    auto tmp1_plain = he_heaan_backend->create_valued_plaintext(99, type);
    he_heaan_backend->decrypt(tmp1_plain, arg0);

    auto tmp1 = dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(tmp1_plain);
    assert(tmp1 != nullptr);

    NGRAPH_INFO << "arg0[0] " << tmp1->m_plaintexts[0];
    NGRAPH_INFO << "arg0[1] " << tmp1->m_plaintexts[1];

    out->m_ciphertext = he_heaan_backend->get_scheme()->multByConstVec(
        arg0->m_ciphertext, tmp, he_heaan_backend->get_precision());

    NGRAPH_INFO << "precision " << he_heaan_backend->get_precision();

    float val_decoded;
    auto tmp_plain = he_heaan_backend->create_valued_plaintext(99, type);
    he_heaan_backend->decrypt(tmp_plain, out);

    auto tmp2 = dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(tmp_plain);
    assert(tmp2 != nullptr);

    NGRAPH_INFO << "result[0] " << tmp2->m_plaintexts[0];
    NGRAPH_INFO << "result[1] " << tmp2->m_plaintexts[1];

    //TODO: reScaleByAndEqual?
}

void runtime::he::kernel::heaan::scalar_multiply(
    const shared_ptr<runtime::he::HeaanPlaintextWrapper>& arg0,
    const shared_ptr<runtime::he::HeaanCiphertextWrapper>& arg1,
    shared_ptr<runtime::he::HeaanCiphertextWrapper>& out,
    const element::Type& type,
    const shared_ptr<runtime::he::he_heaan::HEHeaanBackend> he_heaan_backend)
{
    scalar_multiply(arg1, arg0, out, type, he_heaan_backend);
}

void runtime::he::kernel::heaan::scalar_multiply(
    const shared_ptr<runtime::he::HeaanPlaintextWrapper>& arg0,
    const shared_ptr<runtime::he::HeaanPlaintextWrapper>& arg1,
    shared_ptr<runtime::he::HeaanPlaintextWrapper>& out,
    const element::Type& type,
    const shared_ptr<runtime::he::he_heaan::HEHeaanBackend> he_heaan_backend)
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
    shared_ptr<runtime::he::HEPlaintext> out_he =
        dynamic_pointer_cast<runtime::he::HEPlaintext>(out);
    he_heaan_backend->encode(out_he, &r, type);
    out = dynamic_pointer_cast<runtime::he::HeaanPlaintextWrapper>(out_he);
}
