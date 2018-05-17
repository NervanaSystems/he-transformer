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
    m_cipher_texts.resize(m_num_elements);
    for (size_t i = 0; i < m_num_elements; ++i)
    {
        m_cipher_texts[i] = make_shared<he::HECiphertext>();
    }
}

runtime::he::HECipherTensorView::~HECipherTensorView()
{
}

void runtime::he::HECipherTensorView::write(const void* source, size_t tensor_offset, size_t n)
{
    check_io_bounds(source, tensor_offset, n);
    const element::Type& type = get_tensor_view_layout()->get_element_type();
    size_t type_byte_size = type.size();
    size_t dst_start_index = tensor_offset / type_byte_size;
    size_t num_elements_to_write = n / type_byte_size;

    if (num_elements_to_write == 1)
    {
        const void* src_with_offset = (void*)((char*)source);
        size_t dst_index = dst_start_index;
        he::HEPlaintext p;
        m_he_backend->encode(p, src_with_offset, type);
        m_he_backend->encrypt(*(m_cipher_texts[dst_index]), p);
    }
    else
    {
#pragma omp parallel for
        for (size_t i = 0; i < num_elements_to_write; ++i)
        {
            const void* src_with_offset = (void*)((char*)source + i * type.size());
            size_t dst_index = dst_start_index + i;
            he::HEPlaintext p;
            m_he_backend->encode(p, src_with_offset, type);
            m_he_backend->encrypt(*(m_cipher_texts[dst_index]), p);
        }
    }
}

void runtime::he::HECipherTensorView::read(void* target, size_t tensor_offset, size_t n) const
{
    check_io_bounds(target, tensor_offset, n);
    const element::Type& type = get_tensor_view_layout()->get_element_type();
    size_t type_byte_size = type.size();
    size_t src_start_index = tensor_offset / type_byte_size;
    size_t num_elements_to_read = n / type_byte_size;
    if (num_elements_to_read == 1)
    {
        void* dst_with_offset = (void*)((char*)target);
        size_t src_index = src_start_index;
        he::HEPlaintext p;
        m_he_backend->decrypt(p, *(m_cipher_texts[src_index]));
        m_he_backend->decode(dst_with_offset, p, type);
    }
    else
    {
#pragma omp parallel for
        for (size_t i = 0; i < num_elements_to_read; ++i)
        {
            void* dst_with_offset = (void*)((char*)target + i * type.size());
            size_t src_index = src_start_index + i;
            he::HEPlaintext p;
            m_he_backend->decrypt(p, *(m_cipher_texts[src_index]));
            m_he_backend->decode(dst_with_offset, p, type);
        }
    }
}
