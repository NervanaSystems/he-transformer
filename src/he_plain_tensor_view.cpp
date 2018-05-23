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
#include "he_plain_tensor_view.hpp"
#include "he_seal_backend.hpp"
#include "he_heaan_backend.hpp"
#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "seal_plaintext_wrapper.hpp"
#include "heaan_plaintext_wrapper.hpp"

using namespace ngraph;
using namespace std;

runtime::he::HEPlainTensorView::HEPlainTensorView(const element::Type& element_type,
                                                  const Shape& shape,
                                                  shared_ptr<HEBackend> he_backend,
                                                  const string& name)
    : runtime::he::HETensorView(element_type, shape, he_backend)
{
    // get_tensor_view_layout()->get_size() is the number of elements
    m_num_elements = m_descriptor->get_tensor_view_layout()->get_size();
    m_plain_texts.resize(m_num_elements);
#pragma omp parallel for
    for (size_t i = 0; i < m_num_elements; ++i)
    {
        if (auto he_seal_backend = dynamic_pointer_cast<he_seal::HESealBackend>(m_he_backend))
        {
            m_plain_texts[i] = make_shared<he::SealPlaintextWrapper>();
        }
        else if (auto he_heaan_backend = dynamic_pointer_cast<he_heaan::HEHeaanBackend>(m_he_backend))
        {
            m_plain_texts[i] = make_shared<he::HeaanPlaintextWrapper>();
        }
        else
        {
            throw ngraph_error("m_he_backend neither seal nor heaan in HEPlainTensorView");
        }
    }
}

runtime::he::HEPlainTensorView::~HEPlainTensorView()
{
}

void runtime::he::HEPlainTensorView::write(const void* source, size_t tensor_offset, size_t n)
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

        if (auto he_seal_backend = dynamic_pointer_cast<he_seal::HESealBackend>(m_he_backend))
        {
            he_seal_backend->encode(m_plain_texts[dst_index], src_with_offset, type);
        }
        else if (auto he_heaan_backend = dynamic_pointer_cast<he_heaan::HEHeaanBackend>(m_he_backend))
        {
            NGRAPH_INFO << "HEPlainTensorView::write calling encode";
            he_heaan_backend->encode(m_plain_texts[dst_index], src_with_offset, type);
        }
        else
        {
            throw ngraph_error("HEPlainTensorView::write, he_backend is neither seal nor heaan.");
        }
    }
    else
    {
#pragma omp parallel for
        for (size_t i = 0; i < num_elements_to_write; ++i)
        {
            const void* src_with_offset = (void*)((char*)source + i * type.size());
            size_t dst_index = dst_start_index + i;
            if (auto he_seal_backend = dynamic_pointer_cast<he_seal::HESealBackend>(m_he_backend))
            {
                he_seal_backend->encode(m_plain_texts[dst_index], src_with_offset, type);
            }
            else if (auto he_heaan_backend = dynamic_pointer_cast<he_heaan::HEHeaanBackend>(m_he_backend))
            {
                he_heaan_backend->encode(m_plain_texts[dst_index], src_with_offset, type);
            }
            else
            {
                throw ngraph_error("HEPlainTensorView::write, he_backend is neither seal nor heaan.");
            }
        }
    }
}

void runtime::he::HEPlainTensorView::read(void* target, size_t tensor_offset, size_t n) const
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
        if (auto he_seal_backend = dynamic_pointer_cast<he_seal::HESealBackend>(m_he_backend))
        {
            he_seal_backend->decode(dst_with_offset, m_plain_texts[src_index], type);
        }
        else if (auto he_heaan_backend = dynamic_pointer_cast<he_heaan::HEHeaanBackend>(m_he_backend))
        {
            he_heaan_backend->decode(dst_with_offset, m_plain_texts[src_index], type);
        }
        else
        {
            throw ngraph_error("HEPlainTensorView::read, he_backend is neither seal nor heaan.");
        }
    }
    else
    {
#pragma omp parallel for
        for (size_t i = 0; i < num_elements_to_read; ++i)
        {
            void* dst_with_offset = (void*)((char*)target + i * type.size());
            size_t src_index = src_start_index + i;
            if (auto he_seal_backend = dynamic_pointer_cast<he_seal::HESealBackend>(m_he_backend))
            {
                he_seal_backend->decode(dst_with_offset, m_plain_texts[src_index], type);
            }
            else if (auto he_heaan_backend = dynamic_pointer_cast<he_heaan::HEHeaanBackend>(m_he_backend))
            {
                he_heaan_backend->decode(dst_with_offset, m_plain_texts[src_index], type);
            }
            else
            {
                throw ngraph_error("HEPlainTensorView::read, he_backend is neither seal nor heaan.");
            }
        }
    }
}
