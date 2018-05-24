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
#include "he_seal_backend.hpp"
#include "he_heaan_backend.hpp"

class TestHEBackend : public ::testing::Test
{
protected:
    virtual void SetUp();
    virtual void TearDown();
    // static std::shared_ptr<ngraph::runtime::he::he_seal::HESealBackend> m_he_backend;
    static std::shared_ptr<ngraph::runtime::he::he_heaan::HEHeaanBackend> m_he_backend;
};

std::vector<float> read_constant(const std::string filename);

template <typename T>
bool all_close(const std::vector<std::complex<T>>& a,
               const std::vector<std::complex<T>>& b,
               T atol = static_cast<T>(1e-5))
{
    assert(a.size() == b.size());
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
