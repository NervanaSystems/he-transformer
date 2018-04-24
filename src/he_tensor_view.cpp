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

#include <stack>
#include <vector>

#include "he_backend.hpp"
#include "he_tensor_view.hpp"
#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/descriptor/primary_tensor_view.hpp"
#include "ngraph/runtime/tensor_view.hpp"

using namespace std;
using namespace ngraph;

runtime::he::HETensorView::HETensorView(const element::Type& element_type,
                                        const Shape& shape,
                                        const shared_ptr<HEBackend>& he_backend,
                                        const string& name)
    : runtime::TensorView(make_shared<descriptor::PrimaryTensorView>(
          make_shared<TensorViewType>(element_type, shape), name, true, true, false))
    , m_he_backend(he_backend)
{
    m_descriptor->set_tensor_view_layout(
        make_shared<descriptor::layout::DenseTensorViewLayout>(*m_descriptor));
}

runtime::he::HETensorView::~HETensorView()
{
}

void runtime::he::HETensorView::check_io_bounds(const void* source,
                                                size_t tensor_offset,
                                                size_t n) const
{
    const element::Type& type = get_tensor_view_layout()->get_element_type();
    size_t type_byte_size = type.size();

    // Memory must be byte-aligned to type_byte_size
    // tensor_offset and n are all in bytes
    if (tensor_offset % type_byte_size != 0 || n % type_byte_size != 0)
    {
        std::cout << "tensor_offset " << tensor_offset << endl;
        cout << "type_byte_size " << type_byte_size << endl;
        cout << "n " << n << endl;
        throw ngraph_error("tensor_offset and n must be divisible by type_byte_size.");
    }
    // Check out-of-range
    if ((tensor_offset + n) / type_byte_size > get_element_count())
    {
        throw out_of_range("I/O access past end of tensor");
    }
}

void runtime::he::HETensorView::write(const void* p, size_t tensor_offset, size_t n)
{
    throw ngraph_error("HETensorView write not implemented");
}

void runtime::he::HETensorView::read(void* p, size_t tensor_offset, size_t n) const
{
    throw ngraph_error("HETensorView read not implemented");
}
