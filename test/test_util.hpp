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

#include <complex>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "he_backend.hpp"
#include "he_heaan_backend.hpp"
#include "he_seal_backend.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/node.hpp"

using namespace ngraph;

class TestHEBackend : public ::testing::Test
{
protected:
    virtual void SetUp();
    virtual void TearDown();
    static std::shared_ptr<ngraph::runtime::he::he_seal::HESealBackend> m_he_seal_backend;
    static std::shared_ptr<ngraph::runtime::he::he_heaan::HEHeaanBackend> m_he_heaan_backend;
};

std::vector<float> read_constant(const std::string filename);

template <typename T>
bool all_close(const std::vector<std::complex<T>>& a,
               const std::vector<std::complex<T>>& b,
               T atol = static_cast<T>(1e-5))
{
    for (size_t i = 0; i < a.size(); ++i)
    {
        if ((std::abs(a[i].real() - b[i].real()) > atol) ||
            std::abs(a[i].imag() - b[i].imag()) > atol)
        {
            return false;
        }
    }
    return true;
}

vector<tuple<vector<shared_ptr<ngraph::runtime::TensorView>>,vector<shared_ptr<ngraph::runtime::TensorView>>>>
    generate_plain_cipher_tensors(
    const vector<shared_ptr<Node>>& output, const vector<shared_ptr<Node>>& input,
    shared_ptr<ngraph::runtime::Backend> backend,
    const bool consistent_type = false);
