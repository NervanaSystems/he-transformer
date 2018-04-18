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

#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/descriptor/primary_tensor_view.hpp"
#include "he_plain_tensor_view.hpp"
#include "he_backend.hpp"

using namespace ngraph;
using namespace std;

runtime::he::HEPlainTensorView::HEPlainTensorView(const ngraph::element::Type& element_type,
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
    std::cout << "Creating plain tensor view" << std::endl;
    m_descriptor->set_tensor_view_layout(
        std::make_shared<ngraph::descriptor::layout::DenseTensorViewLayout>(*m_descriptor));

    int plaintext_size = sizeof(seal::Plaintext);
    std::cout << "plaintet size" << plaintext_size << std::endl;
    m_buffer_size = m_descriptor->get_tensor_view_layout()->get_size() * plaintext_size; // element_type.size();

    if (memory_pointer != nullptr)
    {
        m_aligned_buffer_pool = static_cast<char*>(memory_pointer);
    }
    else if (m_buffer_size > 0)
    {
        size_t allocation_size = m_buffer_size + runtime::alignment;
        m_allocated_buffer_pool = static_cast<char*>(malloc(allocation_size));
        m_aligned_buffer_pool = m_allocated_buffer_pool;
        size_t mod = size_t(m_aligned_buffer_pool) % alignment;
        if (mod != 0)
        {
            m_aligned_buffer_pool += (alignment - mod);
        }
    }
    std::cout << "m_buffer_size" << m_buffer_size << std::endl;
}

runtime::he::HEPlainTensorView::HEPlainTensorView(const ngraph::element::Type& element_type,
                                        const Shape& shape,
                                        std::shared_ptr<HEBackend> he_backend,
                                        const string& name)
    : he::HEPlainTensorView(element_type, shape, nullptr, he_backend, name)
{
}

runtime::he::HEPlainTensorView::~HEPlainTensorView()
{
    if (m_allocated_buffer_pool != nullptr)
    {
        free(m_allocated_buffer_pool);
    }
}

char* runtime::he::HEPlainTensorView::get_data_ptr()
{
    return m_aligned_buffer_pool;
}

const char* runtime::he::HEPlainTensorView::get_data_ptr() const
{
    return m_aligned_buffer_pool;
}

void runtime::he::HEPlainTensorView::write(const void* source, size_t tensor_offset, size_t n)
{
    if (tensor_offset + n/sizeof(int) * sizeof(seal::Plaintext) > m_buffer_size)
    {
        throw out_of_range("write access past end of tensor");
    }
    char* target = get_data_ptr();

    size_t offset = tensor_offset;
    int plaintext_size = sizeof(seal::Plaintext);
    int* pt = (int*) source;
    std::cout << "sizeof(pt) " << sizeof(pt) << std::endl;
    std::cout << "sizeof(pt[0]) " << sizeof(pt[0]) << std::endl;
    for(int i = 0; i < n/sizeof(int); ++i) {
        std::cout << "(&target[offset]) " << &target[offset] << std::endl;
        int x = pt[i];
        std::cout << "x " << x << std::endl;
        seal::Plaintext p = (m_he_backend->m_int_encoder)->encode(x);
        std::cout << "p " << p.to_string() << std::endl;
        memcpy(&target[offset], &p, sizeof(seal::Plaintext));

        seal::Plaintext q;

        memcpy(&q, &target[offset], sizeof(seal::Plaintext));
        std::cout << "q " << q.to_string() << std::endl;

        offset += sizeof(seal::Plaintext);
    }
}

void runtime::he::HEPlainTensorView::read(void* target, size_t tensor_offset, size_t n) const
{
    std::cout << "reading " << std::endl;
    if (tensor_offset + n/sizeof(int) * sizeof(seal::Plaintext) > m_buffer_size)
    {
        throw out_of_range("read access past end of tensor");
    }

    //const

    char* source = (char*)(get_data_ptr());
    //seal::Plaintext* pts = (seal::Plaintext*) source;

    size_t offset = tensor_offset;
    for(int i = 0; i < n / sizeof(int); ++i) {
        //seal::Plaintext p = pts[i];
        std::cout << "i " << i << std::endl;
        seal::Plaintext p;
        memcpy(&p, &source[offset], sizeof(seal::Plaintext));
        std::cout << "p " << p.to_string() << std::endl;
        //int x =  (m_he_backend->m_int_encoder)->decode_int64(p);
        //std::cout << "x " << x << std::endl;
        //mempcpy((void*)(target[offset]), &x, sizeof(int));

        offset += sizeof(seal::Plaintext);
    }

    // memcpy(target, &source[tensor_offset], n);
}

size_t runtime::he::HEPlainTensorView::get_size() const
{
    return get_tensor_view_layout()->get_size();
}

const element::Type& runtime::he::HEPlainTensorView::get_element_type() const
{
    return get_tensor_view_layout()->get_element_type();
}
