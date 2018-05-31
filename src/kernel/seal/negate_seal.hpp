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

namespace ngraph
{
    namespace element
    {
        class Type;
    }
    namespace runtime
    {
        namespace he
        {
            class SealCiphertextWrapper;
            class SealPlaintextWrapper;

            namespace he_seal
            {
                class HESealBackend;
            }

            namespace kernel
            {
                namespace seal
                {
                    void
                        scalar_negate(const shared_ptr<runtime::he::SealCiphertextWrapper>& arg0,
                                   shared_ptr<runtime::he::SealCiphertextWrapper>& out,
                                   const element::Type& type,
                                   shared_ptr<runtime::he::he_seal::HESealBackend> he_seal_backend);

                    void
                        scalar_negate(const shared_ptr<runtime::he::SealPlaintextWrapper>& arg0,
                                   shared_ptr<runtime::he::SealPlaintextWrapper>& out,
                                   const element::Type& type,
                                   shared_ptr<runtime::he::he_seal::HESealBackend> he_seal_backend);
                }
            }
        }
    }
}
