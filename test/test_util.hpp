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
#include "he_tensor.hpp"
#include "he_cipher_tensor.hpp"
// #include "seal/ckks/he_seal_ckks_backend.hpp"
#include "seal/bfv/he_seal_bfv_backend.hpp"
#include "ngraph/descriptor/layout/tensor_layout.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/type/element_type.hpp"

using namespace ngraph;

class TestHEBackend : public ::testing::Test
{
protected:
    virtual void SetUp();
    virtual void TearDown();
    static std::shared_ptr<ngraph::runtime::he::he_seal::HESealBFVBackend> m_he_seal_bfv_backend;
};

std::vector<float> read_binary_constant(const std::string filename, size_t num_elements);
std::vector<float> read_constant(const std::string filename);

void write_constant(const std::vector<float>& values, const std::string filename);
void write_binary_constant(const std::vector<float>& values, const std::string filename);

float get_accuracy(const std::vector<float>& pre_sigmoid, const std::vector<float>& y);

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

template <typename T>
bool all_close(const std::vector<T>& a,
               const std::vector<T>& b,
               T atol = static_cast<T>(1e-5))
{
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (std::abs(a[i] - b[i]) > atol)
        {
            return false;
        }
    }
    return true;
}

std::vector<std::tuple<std::vector<std::shared_ptr<ngraph::runtime::Tensor>>,
             std::vector<std::shared_ptr<ngraph::runtime::Tensor>>>>
    generate_plain_cipher_tensors(const std::vector<std::shared_ptr<Node>>& output,
                                  const std::vector<std::shared_ptr<Node>>& input,
                                  std::shared_ptr<ngraph::runtime::Backend> backend,
                                  const bool consistent_type = false);

// Reads batched vector
template <typename T>
std::vector<T> generalized_read_vector(std::shared_ptr<ngraph::runtime::Tensor> tv)
{
    if (ngraph::element::from<T>() != tv->get_tensor_layout()->get_element_type())
    {
        throw std::invalid_argument("read_vector type must match Tensor type");
    }
    if (auto cipher_tv = std::dynamic_pointer_cast<ngraph::runtime::he::HECipherTensor>(tv))
    {
        size_t element_count;
        if (cipher_tv->is_batched())
        {
            element_count = ngraph::shape_size(cipher_tv->get_expanded_shape());
        }
        else
        {
            element_count = ngraph::shape_size(cipher_tv->get_shape());
        }
        size_t size = element_count * sizeof(T);
        std::vector<T> rc(element_count);
        tv->read(rc.data(), 0, size);
        return rc;
    }
    else
    {
        size_t element_count = ngraph::shape_size(tv->get_shape());
        size_t size = element_count * sizeof(T);
        std::vector<T> rc(element_count);
        tv->read(rc.data(), 0, size);
        return rc;
    }
}

template <typename T>
void copy_he_data(std::shared_ptr<ngraph::runtime::Tensor> tv, const std::vector<T>& data, std::shared_ptr<ngraph::runtime::Backend> he_backend)
{
    size_t data_size = data.size() * sizeof(T);
    std::dynamic_pointer_cast<ngraph::runtime::he::HETensor>(tv)->write(data.data(), 0, data_size, std::dynamic_pointer_cast<ngraph::runtime::he::HEBackend>(he_backend).get());
}

template <typename T>
std::vector<T> read_he_vector(std::shared_ptr<ngraph::runtime::Tensor> tv, std::shared_ptr<ngraph::runtime::Backend> he_backend)
{
    if (ngraph::element::from<T>() != tv->get_tensor_layout()->get_element_type())
    {
        throw std::invalid_argument("read_vector type must match Tensor type");
    }
    size_t element_count = ngraph::shape_size(tv->get_shape());
    size_t size = element_count * sizeof(T);
    std::vector<T> rc(element_count);
    std::dynamic_pointer_cast<ngraph::runtime::he::HETensor>(tv)->read(rc.data(), 0, size, std::dynamic_pointer_cast<ngraph::runtime::he::HEBackend>(he_backend).get());
    return rc;
}

