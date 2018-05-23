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

#include <vector>

namespace ngraph
{
    namespace runtime
    {
        namespace he
        {
            template <typename S, typename T>
                bool cast_vector(std::vector<std::shared_ptr<T>>& output, const std::vector<std::shared_ptr<S>>& input)
                {
					assert(output.size() == input.size());
					for(size_t i = 0; i < input.size(); ++i)
					{
						output[i] = dynamic_pointer_cast<T>(input[i]);
						if (output[i] == nullptr)
						{
							return false;
						}
					}
					return true;

                }
        }
    }
}

