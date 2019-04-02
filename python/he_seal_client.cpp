//*****************************************************************************
// Copyright 2018-2019 Intel Corporation
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

#include <iterator>
#include <sstream>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "he_seal_client.hpp"
#include "seal/he_seal_client.hpp"

namespace py = pybind11;

void regclass_pyhetransformer_HESealClient(py::module m) {
  py::class_<ngraph::runtime::he::HESealClient> he_seal_client(m,
                                                               "HESealClient");
  he_seal_client.doc() = "he_seal_client doc";

  he_seal_client.def(
      py::init<boost::asio::io_context&, const tcp::resolver::results_type&,
               std::vector<float>>());
}
