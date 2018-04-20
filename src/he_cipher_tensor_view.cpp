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

#include <cstring>
#include <memory>

#include "he_backend.hpp"
#include "he_cipher_tensor_view.hpp"
#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/descriptor/primary_tensor_view.hpp"

using namespace ngraph;
using namespace std;

runtime::he::HECipherTensorView::HECipherTensorView(const ngraph::element::Type& element_type,
                                                    const Shape& shape,
                                                    void* memory_pointer,
                                                    std::shared_ptr<HEBackend> he_backend,
                                                    const string& name)
    : runtime::he::HETensorView(std::make_shared<ngraph::descriptor::PrimaryTensorView>(
          std::make_shared<ngraph::TensorViewType>(element_type, shape), name, true, true, false))
    , m_allocated_buffer_pool(nullptr)
    , m_aligned_buffer_pool(nullptr)
    , m_he_backend(he_backend)
{
    m_descriptor->set_tensor_view_layout(
        std::make_shared<ngraph::descriptor::layout::DenseTensorViewLayout>(*m_descriptor));

    int ciphertext_size = sizeof(seal::Ciphertext);
    m_buffer_size = m_descriptor->get_tensor_view_layout()->get_size() * ciphertext_size;

    if (memory_pointer != nullptr)
    {
        m_aligned_buffer_pool = static_cast<seal::Ciphertext*>(memory_pointer);
    }
    else if (m_buffer_size > 0)
    {
        size_t allocation_size = m_buffer_size + runtime::alignment;
        m_allocated_buffer_pool = static_cast<seal::Ciphertext*>(malloc(allocation_size));
        m_aligned_buffer_pool = m_allocated_buffer_pool;
        size_t mod = size_t(m_aligned_buffer_pool) % alignment;
        if (mod != 0)
        {
            m_aligned_buffer_pool += (alignment - mod);
        }
    }
}

runtime::he::HECipherTensorView::HECipherTensorView(const ngraph::element::Type& element_type,
                                                    const Shape& shape,
                                                    std::shared_ptr<HEBackend> he_backend,
                                                    const string& name)
    : he::HECipherTensorView(element_type, shape, nullptr, he_backend, name)
{
}

runtime::he::HECipherTensorView::~HECipherTensorView()
{
    if (m_allocated_buffer_pool != nullptr)
    {
        free(m_allocated_buffer_pool);
    }
}

seal::Ciphertext* runtime::he::HECipherTensorView::get_data_ptr()
{
    return m_aligned_buffer_pool;
}

const seal::Ciphertext* runtime::he::HECipherTensorView::get_data_ptr() const
{
    return m_aligned_buffer_pool;
}

void runtime::he::HECipherTensorView::write(const void* source, size_t tensor_offset, size_t n)
{
    const element::Type& type = get_element_type();
    if (tensor_offset + n / type.size() * sizeof(seal::Ciphertext) > m_buffer_size)
    {
        throw out_of_range("write access past end of tensor");
    }
    seal::Ciphertext* target = get_data_ptr();

    size_t offset = tensor_offset / sizeof(seal::Ciphertext);

    for (int i = 0; i < n / type.size(); ++i)
    {
        seal::Plaintext* p = new seal::Plaintext;
        seal::Ciphertext* c = new seal::Ciphertext;
        m_he_backend->encode(p, (void*)((char*)source + i * type.size()), type);
        m_he_backend->encrypt(*c, *p);
        memcpy(&target[offset + i], c, sizeof(seal::Ciphertext));
    }
}

void runtime::he::HECipherTensorView::read(void* target, size_t tensor_offset, size_t n) const
{
    const element::Type& type = get_element_type();
    if (tensor_offset + n / type.size() * sizeof(seal::Ciphertext) > m_buffer_size)
    {
        throw out_of_range("read access past end of tensor");
    }
    const seal::Ciphertext* source = get_data_ptr();

    size_t offset = tensor_offset / sizeof(seal::Ciphertext);
    for (int i = 0; i < n / type.size(); ++i)
    {
        const seal::Ciphertext c = source[i + offset];
        seal::Plaintext* p = new seal::Plaintext;
        m_he_backend->decrypt(*p, c);
        m_he_backend->decode((void*)((char*)target + i * type.size()), *p, type);
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
