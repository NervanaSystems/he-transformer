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

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "ngraph/log.hpp"
#include "nlohmann/json.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace he
        {
            class HEHeaanParameter
            {
            public:
                HEHeaanParameter(std::uint64_t log2_poly_modulus,
                                 std::uint64_t log2_plain_modulus,
                                 std::uint64_t log2_precision)
                    : m_log2_poly_modulus(log2_poly_modulus)
                    , m_log2_plain_modulus(log2_plain_modulus)
                    , m_log2_precision(log2_precision)
                {
                }
                ~HEHeaanParameter() {}
                std::uint64_t m_log2_poly_modulus;
                std::uint64_t m_log2_plain_modulus;
                std::uint64_t m_log2_precision;
            };

            const static HEHeaanParameter parse_heaan_config_or_use_default()
            {
                try
                {
                    const char* config_path = std::getenv("NGRAPH_HE_HEAAN_CONFIG");
                    if (config_path != nullptr)
                    {
                        // Read file to string
                        std::ifstream f(config_path);
                        std::stringstream ss;
                        ss << f.rdbuf();
                        std::string s = ss.str();

                        // Parse json
                        nlohmann::json js = nlohmann::json::parse(s);
                        std::uint64_t log2_poly_modulus = js["log2_poly_modulus"];
                        std::uint64_t log2_plain_modulus = js["log2_plain_modulus"];
                        std::uint64_t log2_precision = js["log2_precision"];

                        NGRAPH_INFO << "Using HEAAN config for parameters: " << config_path;
                        return HEHeaanParameter(
                            log2_poly_modulus, log2_plain_modulus, log2_precision);
                    }
                    else
                    {
                        NGRAPH_INFO << "Using HEAAN default parameters" << config_path;
                        throw std::runtime_error("config_path is NULL");
                    }
                }
                catch (const std::exception& e)
                {
                    return HEHeaanParameter(13, 383, 32);
                }
            }
            static HEHeaanParameter default_heaan_parameter = parse_heaan_config_or_use_default();
        }
    }
}
