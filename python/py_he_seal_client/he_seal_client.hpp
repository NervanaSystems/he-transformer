//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include <iterator>
#include <sstream>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <boost/asio.hpp>

#include "he_seal_client.hpp"
#include "seal/he_seal_client.hpp"

namespace py = pybind11;

PYBIND11_MODULE(he_seal_client, m) {
  py::class_<ngraph::runtime::he::HESealClient> he_seal_client(m,
                                                               "HESealClient");
  he_seal_client.doc() =
      "he_seal_client wraps ngraph::runtime::he::HESealClient";

  he_seal_client.def(py::init<const std::string&, const std::size_t,
                              const std::vector<float>&>());

  he_seal_client.def("set_seal_context",
                     &ngraph::runtime::he::HESealClient::set_seal_context);
  he_seal_client.def("handle_message",
                     &ngraph::runtime::he::HESealClient::handle_message);
  he_seal_client.def("write_message",
                     &ngraph::runtime::he::HESealClient::write_message);
  he_seal_client.def("is_done", &ngraph::runtime::he::HESealClient::is_done);
  he_seal_client.def("get_results",
                     &ngraph::runtime::he::HESealClient::get_results);
  he_seal_client.def("close_connection",
                     &ngraph::runtime::he::HESealClient::close_connection);
}
