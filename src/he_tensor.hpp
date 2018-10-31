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

#pragma once

#include <memory>

#include "ngraph/runtime/tensor.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace he
        {
            class HEBackend;

            class HETensor : public runtime::Tensor
            {
            public:
                HETensor(const element::Type& element_type,
                         const Shape& shape,
                         const HEBackend* he_backend,
                         bool batched = false,
                         const std::string& name = "external");
                virtual ~HETensor(){};

                /// @brief Write bytes directly into the tensor
                /// @param p Pointer to source of data
                /// @param tensor_offset Offset into tensor storage to begin writing. Must be element-aligned.
                /// @param n Number of bytes to write, must be integral number of elements.
                virtual void write(const void* p, size_t tensor_offset, size_t n) override = 0;

                /// @brief Read bytes directly from the tensor
                /// @param p Pointer to destination for data
                /// @param tensor_offset Offset into tensor storage to begin reading. Must be element-aligned.
                /// @param n Number of bytes to read, must be integral number of elements.
                virtual void read(void* p, size_t tensor_offset, size_t n) const override = 0;

                /// @brief Reduces shape along batch axis
                /// @param shape Input shape to batch
                /// @param batch_dim Axis along which to batch
                /// @param batched Whether or not batching is enabled
                /// @return Shape after batching along batch axis
                const Shape batch_shape(const Shape& shape,
                                        size_t batch_axis = 0,
                                        bool batched = false) const;

            protected:
                void check_io_bounds(const void* p, size_t tensor_offset, size_t n) const;

                bool m_batched;
                // TODO: support more arbitrary batching dimension
                size_t m_batch_size; // If m_batched, corresponds to first shape dimesion.

                const HEBackend* m_he_backend;
            };
        }
    }
}
