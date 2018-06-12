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
#include "he_heaan_backend.hpp"
#include "he_seal_backend.hpp"
#include "heaan_ciphertext_wrapper.hpp"
#include "heaan_plaintext_wrapper.hpp"
#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/util.hpp"
#include "seal_ciphertext_wrapper.hpp"
#include "seal_plaintext_wrapper.hpp"

using namespace ngraph;
using namespace std;

runtime::he::HECipherTensorView::HECipherTensorView(const element::Type& element_type,
                                                    const Shape& shape,
                                                    shared_ptr<HEBackend> he_backend,
                                                    const bool batched,
                                                    const string& name)
    : runtime::he::HETensorView(element_type, shape, he_backend, batched, name)
{
    // get_tensor_view_layout()->get_size() is the number of elements
    m_num_elements = m_descriptor->get_tensor_view_layout()->get_size();
    m_cipher_texts.resize(m_num_elements);
    for (size_t i = 0; i < m_num_elements; ++i)
    {
        if (auto he_seal_backend = dynamic_pointer_cast<he_seal::HESealBackend>(m_he_backend))
        {
            m_cipher_texts[i] = make_shared<runtime::he::SealCiphertextWrapper>();
        }
        else if (auto he_heaan_backend =
                     dynamic_pointer_cast<he_heaan::HEHeaanBackend>(m_he_backend))
        {
            m_cipher_texts[i] = make_shared<runtime::he::HeaanCiphertextWrapper>(m_batch_size);
        }
        else
        {
            throw ngraph_error(
                "HECipherTensorView::HECipherTensorView(), he_backend is neither SEAL nor HEAAN.");
        }
    }
}

runtime::he::HECipherTensorView::~HECipherTensorView()
{
}

const Shape runtime::he::HECipherTensorView::get_expanded_shape() const
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

void runtime::he::HECipherTensorView::write(const void* source, size_t tensor_offset, size_t n)
{
    check_io_bounds(source, tensor_offset, n / m_batch_size);
    const element::Type& type = get_tensor_view_layout()->get_element_type();
    size_t type_byte_size = type.size();
    size_t dst_start_index = tensor_offset / type_byte_size;
    size_t num_elements_to_write = n / (type_byte_size * m_batch_size);

    if (num_elements_to_write == 1)
    {
        const void* src_with_offset = (void*)((char*)source);
        size_t dst_index = dst_start_index;

        if (auto he_seal_backend = dynamic_pointer_cast<he_seal::HESealBackend>(m_he_backend))
        {
            shared_ptr<runtime::he::HEPlaintext> p =
                make_shared<runtime::he::SealPlaintextWrapper>();
            he_seal_backend->encode(p, src_with_offset, type);
            he_seal_backend->encrypt(m_cipher_texts[dst_index], p);
        }
        else if (auto he_heaan_backend =
                     dynamic_pointer_cast<he_heaan::HEHeaanBackend>(m_he_backend))
        {
            shared_ptr<runtime::he::HEPlaintext> p =
                make_shared<runtime::he::HeaanPlaintextWrapper>();
            he_heaan_backend->encode(p, src_with_offset, type, m_batch_size);
            he_heaan_backend->encrypt(m_cipher_texts[dst_index], p);
        }
        else
        {
            throw ngraph_error("HECipherTensorView::write, he_backend is neither SEAL nor HEAAN.");
        }
    }
    else
    {
#pragma omp parallel for
        for (size_t i = 0; i < num_elements_to_write; ++i)
        {
            const void* src_with_offset = (void*)((char*)source + i * type.size() * m_batch_size);
            size_t dst_index = dst_start_index + i;

            if (auto he_seal_backend = dynamic_pointer_cast<he_seal::HESealBackend>(m_he_backend))
            {
                shared_ptr<runtime::he::HEPlaintext> p =
                    make_shared<runtime::he::SealPlaintextWrapper>();
                he_seal_backend->encode(p, src_with_offset, type);
                he_seal_backend->encrypt(m_cipher_texts[dst_index], p);
            }
            else if (auto he_heaan_backend =
                         dynamic_pointer_cast<he_heaan::HEHeaanBackend>(m_he_backend))
            {
                shared_ptr<runtime::he::HEPlaintext> p =
                    make_shared<runtime::he::HeaanPlaintextWrapper>();
                if (m_batched)
                {
                    size_t allocation_size = type.size() * m_batch_size;
                    const void* batch_src = malloc(allocation_size);
                    if (!batch_src)
                    {
                        throw ngraph_error("Error allocating HE Cipher Tensor View memory");
                    }

                    for (size_t j = 0; j < m_batch_size; ++j)
                    {
                        void* destination = (void*)((char*)batch_src + j * type.size());
                        const void* src =
                            (void*)((char*)source + type.size() * (i + j * num_elements_to_write));
                        memcpy(destination, src, type.size());
                    }

                    he_heaan_backend->encode(p, batch_src, type, m_batch_size);
                    free((void*)batch_src);
                }
                else
                {
                    he_heaan_backend->encode(p, src_with_offset, type, m_batch_size);
                }
                he_heaan_backend->encrypt(m_cipher_texts[dst_index], p);
            }
            else
            {
                throw ngraph_error(
                    "HECipherTensorView::write, he_backend is neither SEAL nor HEAAN.");
            }
        }
    }
}

void runtime::he::HECipherTensorView::read(void* target, size_t tensor_offset, size_t n) const
{
    check_io_bounds(target, tensor_offset, n / m_batch_size);
    const element::Type& type = get_tensor_view_layout()->get_element_type();
    size_t type_byte_size = type.size();
    size_t src_start_index = tensor_offset / type_byte_size;
    size_t num_elements_per_batch = n / (type_byte_size * m_batch_size);
    size_t num_elements_to_read = n / (type_byte_size * m_batch_size);

    if (num_elements_to_read == 1)
    {
        void* dst_with_offset = (void*)((char*)target);
        size_t src_index = src_start_index;
        if (auto he_seal_backend = dynamic_pointer_cast<he_seal::HESealBackend>(m_he_backend))
        {
            shared_ptr<runtime::he::HEPlaintext> p =
                make_shared<runtime::he::SealPlaintextWrapper>();
            he_seal_backend->decrypt(p, m_cipher_texts[src_index]);
            he_seal_backend->decode(dst_with_offset, p, type);
        }
        else if (auto he_heaan_backend =
                     dynamic_pointer_cast<he_heaan::HEHeaanBackend>(m_he_backend))
        {
            shared_ptr<runtime::he::HEPlaintext> p =
                make_shared<runtime::he::HeaanPlaintextWrapper>();
            he_heaan_backend->decrypt(p, m_cipher_texts[src_index]);
            he_heaan_backend->decode(dst_with_offset, p, type, m_batch_size);
        }
        else
        {
            throw ngraph_error("HECipherTensorView::read he_backend is neither SEAL nor HEAAN.");
        }
    }
    else
    {
#pragma omp parallel for
        for (size_t i = 0; i < num_elements_to_read; ++i)
        {
            void* dst = malloc(type.size() * m_batch_size);
            if (!dst)
            {
                throw ngraph_error("Error allocating HE Cipher Tensor View memory");
            }

            size_t src_index = src_start_index + i;
            if (auto he_seal_backend = dynamic_pointer_cast<he_seal::HESealBackend>(m_he_backend))
            {
                shared_ptr<runtime::he::HEPlaintext> p =
                    make_shared<runtime::he::SealPlaintextWrapper>();
                he_seal_backend->decrypt(p, m_cipher_texts[src_index]);
                he_seal_backend->decode(dst, p, type);
            }
            else if (auto he_heaan_backend =
                         dynamic_pointer_cast<he_heaan::HEHeaanBackend>(m_he_backend))
            {
                shared_ptr<runtime::he::HEPlaintext> p =
                    make_shared<runtime::he::HeaanPlaintextWrapper>();
                he_heaan_backend->decrypt(p, m_cipher_texts[src_index]);
                he_heaan_backend->decode(dst, p, type, m_batch_size);
            }
            else
            {
                throw ngraph_error("HECipherTensorView::read he_backend is not seal.");
            }
            for (size_t j = 0; j < m_batch_size; ++j)
            {
                void* dst_with_offset =
                    (void*)((char*)target + type.size() * (i + j * num_elements_to_read));
                const void* src = (void*)((char*)dst + j * type.size());
                memcpy(dst_with_offset, src, type.size());
            }
            free(dst);
        }
    }
}
