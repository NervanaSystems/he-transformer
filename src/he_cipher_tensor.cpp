//*****************************************************************************
// Copyright 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <cstring>
#include <memory>
#include <string>

#include "he_backend.hpp"
#include "he_cipher_tensor.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

runtime::he::HECipherTensor::HECipherTensor(const element::Type& element_type,
                                            const Shape& shape,
                                            const HEBackend* he_backend,
                                            const std::shared_ptr<HECiphertext> he_ciphertext,
                                            const bool batched,
                                            const string& name)
    : runtime::he::HETensor(element_type, shape, he_backend, batched, name)
{
    m_num_elements = m_descriptor->get_tensor_layout()->get_size();
    m_cipher_texts.resize(m_num_elements);
    for (size_t i = 0; i < m_num_elements; ++i)
    {
        m_cipher_texts[i] = he_backend->create_empty_ciphertext();
    }
}

const Shape runtime::he::HECipherTensor::get_expanded_shape() const
{
    if (m_batched)
    {
        Shape expanded_shape = get_shape();
        if (is_scalar(expanded_shape))
        {
            return Shape{m_batch_size, 1};
        }
        else if (expanded_shape[0] == 1)
        {
            expanded_shape[0] = m_batch_size;
        }
        else
        {
            expanded_shape.insert(expanded_shape.begin(), m_batch_size);
        }
        return expanded_shape;
    }
    else
    {
        return get_shape();
    }
}

void runtime::he::HECipherTensor::write(const void* source, size_t tensor_offset, size_t n)
{
    // Hack to fix Cryptonets with ngraph-tf
    const char* ng_batch_tensor_value = std::getenv("NGRAPH_BATCHED_TENSOR");
    if (ng_batch_tensor_value != nullptr)
    {
        n *= m_batch_size;
    }

    check_io_bounds(source, tensor_offset, n / m_batch_size);
    const element::Type& element_type = get_tensor_layout()->get_element_type();
    size_t type_byte_size = element_type.size();
    size_t dst_start_index = tensor_offset / type_byte_size;
    size_t num_elements_to_write = n / (type_byte_size * m_batch_size);

    if (num_elements_to_write == 1)
    {
        const void* src_with_offset = (void*)((char*)source);
        size_t dst_index = dst_start_index;

        shared_ptr<runtime::he::HEPlaintext> plaintext = m_he_backend->create_empty_plaintext();
        m_he_backend->encode(plaintext, src_with_offset, element_type, m_batch_size);
        m_he_backend->encrypt(m_cipher_texts[dst_index], plaintext);
    }
    else
    {
#pragma omp parallel for
        for (size_t i = 0; i < num_elements_to_write; ++i)
        {

            const void* src_with_offset =
                (void*)((char*)source + i * type_byte_size * m_batch_size);
            size_t dst_index = dst_start_index + i;

            shared_ptr<runtime::he::HEPlaintext> plaintext = m_he_backend->create_empty_plaintext();
            if (m_batch_size > 1)
            {
                size_t allocation_size = type_byte_size * m_batch_size;
                const void* batch_src = malloc(allocation_size);
                if (!batch_src)
                {
                    throw ngraph_error("Error allocating HE Cipher Tensor View memory");
                }

                for (size_t j = 0; j < m_batch_size; ++j)
                {
                    void* destination = (void*)((char*)batch_src + j * type_byte_size);
                    const void* src =
                        (void*)((char*)source + type_byte_size * (i + j * num_elements_to_write));
                    memcpy(destination, src, type_byte_size);
                }

                m_he_backend->encode(plaintext, batch_src, element_type, m_batch_size);
                free((void*)batch_src);
            }
            else
            {
                m_he_backend->encode(plaintext, src_with_offset, element_type, m_batch_size);
            }
            m_he_backend->encrypt(m_cipher_texts[dst_index], plaintext);

        }
    }
}

void runtime::he::HECipherTensor::read(void* target, size_t tensor_offset, size_t n) const
{
    // Hack to fix Cryptonets with ngraph-tf
    const char* ng_batch_tensor_value = std::getenv("NGRAPH_BATCHED_TENSOR");
    if (ng_batch_tensor_value != nullptr)
    {
        n *= m_batch_size;
    }

    check_io_bounds(target, tensor_offset, n / m_batch_size);
    const element::Type& element_type = get_tensor_layout()->get_element_type();
    size_t type_byte_size = element_type.size();
    size_t src_start_index = tensor_offset / type_byte_size;
    size_t num_elements_to_read = n / (type_byte_size * m_batch_size);

    if (num_elements_to_read == 1)
    {
        void* dst_with_offset = (void*)((char*)target);
        size_t src_index = src_start_index;
        shared_ptr<runtime::he::HEPlaintext> p = m_he_backend->create_empty_plaintext();
        m_he_backend->decrypt(p, m_cipher_texts[src_index]);
        m_he_backend->decode(dst_with_offset, p, element_type, m_batch_size);
    }
    else
    {
#pragma omp parallel for
        for (size_t i = 0; i < num_elements_to_read; ++i)
        {
            void* dst = malloc(type_byte_size * m_batch_size);
            if (!dst)
            {
                throw ngraph_error("Error allocating HE Cipher Tensor memory");
            }

            size_t src_index = src_start_index + i;
            shared_ptr<runtime::he::HEPlaintext> p = m_he_backend->create_empty_plaintext();
            m_he_backend->decrypt(p, m_cipher_texts[src_index]);
            m_he_backend->decode(dst, p, element_type, m_batch_size);

            for (size_t j = 0; j < m_batch_size; ++j)
            {
                void* dst_with_offset =
                    (void*)((char*)target + type_byte_size * (i + j * num_elements_to_read));
                const void* src = (void*)((char*)dst + j * type_byte_size);
                memcpy(dst_with_offset, src, type_byte_size);
            }
            free(dst);
        }
    }
}
