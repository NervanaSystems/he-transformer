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

#include <memory>
#include <string>

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
    // get_tensor_view_layout()->get_size() is the number of elements
    m_num_elements = m_descriptor->get_tensor_view_layout()->get_size();
    if (m_num_elements > 0)
    {
        m_cipher_texts = vector<shared_ptr<seal::Ciphertext>>(m_num_elements);
    }
}

runtime::he::HECipherTensorView::HECipherTensorView(const element::Type& element_type,
                                                    const Shape& shape,
                                                    void* memory_pointer,
                                                    shared_ptr<HEBackend> he_backend,
                                                    const string& name)
    : runtime::he::HETensorView(element_type, shape, he_backend)
{
    throw ngraph_error("HECipherTensorView with predefined host memory_pointer is not supported");
}

runtime::he::HECipherTensorView::~HECipherTensorView()
{
}

void runtime::he::HECipherTensorView::write(const void* source, size_t tensor_offset, size_t n)
{
    const element::Type& type = get_element_type();
    size_t type_byte_size = type.size();

    // Memory must be byte-aligned to type_byte_size
    // tensor_offset and n are all in bytes
    if (tensor_offset % type_byte_size != 0 || n % type_byte_size != 0)
    {
        throw ngraph_error("tensor_offset and n must be divisable by type_byte_size.");
    }
    // Check out-of-range
    if ((tensor_offset + n) / type_byte_size > m_num_elements)
    {
        throw out_of_range("Write access past end of tensor");
    }

    size_t start_index = tensor_offset / type_byte_size; // inclusive
    size_t num_elements_to_write = n / type_byte_size;
    for (size_t i = 0; i < num_elements_to_write; ++i)
    {
        const void* src_with_offset = (void*)((char*)source + i * type.size());
        size_t dst_index = start_index + i;

        seal::Plaintext p;
        m_he_backend->encode(p, src_with_offset, type);
        // m_he_backend->encrypt(*(m_cipher_texts[dst_index]), p);
    }


    // for (size_t i = start_index; i < end_index; ++i)
    // {
    //     void* source_with_offset = (static_cast<char*>source)
    //     seal::Plaintext p;
    //     seal::Ciphertext c;
    //     m_he_backend->encode(p, (void*)((char*)source + i * type.size()), type);
    //     // m_he_backend->encrypt(c, p);
    //     m_he_backend->encrypt(*(m_cipher_texts[0]), p);
    // }


    // for (int i = 0; i < n / type.size(); ++i)
    // {
    //     seal::Plaintext p;
    //     seal::Ciphertext c;
    //     m_he_backend->encode(p, (void*)((char*)source + i * type.size()), type);
    //     m_he_backend->encrypt(c, p);
    //     m_cipher_texts[offset + i] = make_shared<seal::Ciphertext>(c);
    // }
}

void runtime::he::HECipherTensorView::read(void* target, size_t tensor_offset, size_t n) const
{
    // const element::Type& type = get_element_type();
    // if (tensor_offset / sizeof(seal::Ciphertext) + n / type.size() > m_num_elements)
    // {
    //     throw out_of_range("read access past end of tensor");
    // }

    // const vector<shared_ptr<seal::Ciphertext>>& source = m_cipher_texts;

    // size_t offset = tensor_offset / sizeof(seal::Ciphertext);
    // for (int i = 0; i < n / type.size(); ++i)
    // {
    //     const shared_ptr<seal::Ciphertext> c = source[offset + i];
    //     seal::Plaintext p;
    //     m_he_backend->decrypt(p, *c);
    //     m_he_backend->decode((void*)((char*)target + i * type.size()), p, type);
    // }
}

size_t runtime::he::HECipherTensorView::get_size() const
{
    return get_tensor_view_layout()->get_size();
}

const element::Type& runtime::he::HECipherTensorView::get_element_type() const
{
    return get_tensor_view_layout()->get_element_type();
}
