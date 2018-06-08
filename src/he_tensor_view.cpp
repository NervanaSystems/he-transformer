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

// #include <stack>

#include "he_backend.hpp"
#include "he_tensor_view.hpp"
#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/descriptor/primary_tensor_view.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

runtime::he::HETensorView::HETensorView(const element::Type& element_type,
                                        const Shape& shape,
                                        const shared_ptr<HEBackend>& he_backend,
                                        bool batched,
                                        const string& name)
    : runtime::TensorView(make_shared<descriptor::PrimaryTensorView>(
          make_shared<TensorViewType>(element_type, batch_shape(shape, 0, batched)), name))
{
    m_descriptor->set_tensor_view_layout(
        make_shared<descriptor::layout::DenseTensorViewLayout>(*m_descriptor));
    auto is_power_of_2 = [](size_t n) -> bool { return ((n & (n - 1)) == 0) && (n != 0); };

    if (batched)
    {
        if (!is_power_of_2(shape[0]))
        {
            throw ngraph_error("Batching size must be power of two");
        }

        m_batch_size = shape[0];
    }
    else
    {
        m_batch_size = 1;
    }
    NGRAPH_INFO << "Creating HETV with m_batch_size " << m_batch_size;
    m_he_backend = he_backend;
    m_batched = batched;
}

runtime::he::HETensorView::~HETensorView()
{
}

const Shape runtime::he::HETensorView::batch_shape(const Shape& shape,
                                                   size_t batch_axis,
                                                   bool batched) const
{
    if (batched)
    {
        if (batch_axis != 0)
        {
            throw ngraph_error("Batching only supported along axis 0");
        }
        Shape ret(shape);
        ret[batch_axis] = 1;
        NGRAPH_INFO << "Batching shape " << join(shape, "x") << " to " << join(ret, "x");

        return ret;
    }
    return shape;
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
        NGRAPH_INFO << "tensor_offset " << tensor_offset;
        NGRAPH_INFO << "type_byte_size " << type_byte_size;
        NGRAPH_INFO << "n " << n;
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
