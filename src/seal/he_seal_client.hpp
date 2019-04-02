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

#include <boost/asio.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "seal/he_seal_client.hpp"
#include "seal/he_seal_util.hpp"
#include "seal/seal.h"
#include "tcp/tcp_client.hpp"
#include "tcp/tcp_message.hpp"

using namespace ngraph;

ngraph::runtime::he::HESealClient::HESealClient(
    boost::asio::io_context& io_context,
    const tcp::resolver::results_type& endpoints, std::vector<float> inputs);

void ngraph::runtime::he::HESealClient::set_seal_context();

void ngraph::runtime::he::HESealClient::handle_message(
    const runtime::he::TCPMessage& message);

void ngraph::runtime::he::HESealClient::write_message(
    const runtime::he::TCPMessage& message);

bool ngraph::runtime::he::HESealClient::is_done();

std::vector<float> ngraph::runtime::he::HESealClient::get_results();

void ngraph::runtime::he::HESealClient::close_connection();
