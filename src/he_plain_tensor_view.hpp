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

#pragma once

#include <string>
#include <vector>

#include "he_plaintext.hpp"
#include "he_tensor_view.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace he
        {
            class HEBackend;
            class HEPlaintext;

            class HEPlainTensorView : public HETensorView
            {
            public:
                HEPlainTensorView(const element::Type& element_type,
                                  const Shape& shape,
                                  std::shared_ptr<HEBackend> he_backend,
                                  const std::string& name = "external");
                virtual ~HEPlainTensorView();

                /// @brief Write bytes directly into the tensor after encoding
                /// @param p Pointer to source of data
                /// @param tensor_offset Offset (bytes) into tensor storage to begin writing.
                ///        Must be element-aligned.
                /// @param n Number of bytes to write, must be integral number of elements.
                void write(const void* p, size_t tensor_offset, size_t n);

                /// @brief Read bytes directly from the tensor after decoding
                /// @param p Pointer to destination for data
                /// @param tensor_offset Offset (bytes) into tensor storage to begin reading.
                ///        Must be element-aligned.
                /// @param n Number of bytes to read, must be integral number of elements.
                void read(void* p, size_t tensor_offset, size_t n) const;

                inline std::vector<std::shared_ptr<runtime::he::HEPlaintext>>& get_elements()
                {
                    return m_plain_texts;
                }

                inline std::shared_ptr<runtime::he::HEPlaintext>& get_element(size_t i)
                {
                    return m_plain_texts[i];
                }

            private:
                std::vector<std::shared_ptr<runtime::he::HEPlaintext>> m_plain_texts;
                size_t m_num_elements;
            };
        }
    }
}
