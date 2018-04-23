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

#include <string>
#include <memory>

#include "he_backend.hpp"
#include "he_cipher_tensor_view.hpp"
#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"

using namespace ngraph;
using namespace std;

runtime::he::HECipherTensorView::HECipherTensorView(const element::Type& element_type,
                                                    const Shape& shape,
                                                    shared_ptr<HEBackend> he_backend,
                                                    const string& name)
    : runtime::he::HETensorView(element_type, shape, he_backend)
{
    m_buffer_size = m_descriptor->get_tensor_view_layout()->get_size();

    if (m_buffer_size > 0)
    {
        m_allocated_buffer_pool = vector<shared_ptr<seal::Ciphertext>>(m_buffer_size);
    }
}

runtime::he::HECipherTensorView::~HECipherTensorView()
{
}

void runtime::he::HECipherTensorView::write(const void* source, size_t tensor_offset, size_t n)
{
    const element::Type& type = get_element_type();
    if (tensor_offset / sizeof(seal::Ciphertext) + n / type.size() > m_buffer_size)
    {
        throw out_of_range("write access past end of tensor");
    }
    vector<shared_ptr<seal::Ciphertext>>& target = m_allocated_buffer_pool;

    size_t offset = tensor_offset / sizeof(seal::Ciphertext);
    for (int i = 0; i < n / type.size(); ++i)
    {
        seal::Plaintext p;
        seal::Ciphertext c;
        m_he_backend->encode(p, (void*)((char*)source + i * type.size()), type);
        m_he_backend->encrypt(c, p);

        target[offset + i] = make_shared<seal::Ciphertext>(c);
    }
}

void runtime::he::HECipherTensorView::read(void* target, size_t tensor_offset, size_t n) const
{
    const element::Type& type = get_element_type();
    if (tensor_offset / sizeof(seal::Ciphertext) + n / type.size() > m_buffer_size)
    {
        throw out_of_range("read access past end of tensor");
    }

    const vector<shared_ptr<seal::Ciphertext>>& source = m_allocated_buffer_pool;

    size_t offset = tensor_offset / sizeof(seal::Ciphertext);
    for (int i = 0; i < n / type.size(); ++i)
    {
        const shared_ptr<seal::Ciphertext> c = source[offset + i];
        seal::Plaintext p;
        m_he_backend->decrypt(p, *c);
        m_he_backend->decode((void*)((char*)target + i * type.size()), p, type);
    }
}

size_t runtime::he::HECipherTensorView::get_size() const
{
    return get_tensor_view_layout()->get_size();
}

const element::Type& runtime::he::HECipherTensorView::get_element_type() const
{
    return get_tensor_view_layout()->get_element_type();
}
