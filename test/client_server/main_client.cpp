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

#include <iostream>
#include <vector>
#include "seal/he_seal_client.hpp"

using namespace ngraph;

int main() {
  std::vector<float> inputs{1, 2, 3, 4, 5};

  std::string hostname = "localhost";
  std::size_t port = 34000;
  boost::asio::io_context io_context;
  tcp::resolver resolver(io_context);
  auto client = runtime::he::HESealClient(hostname, port, inputs);

  while (!client.is_done()) {
    sleep(1);
  }
  std::vector<float> results = client.get_results();

  std::cout << "Result " << std::endl;
  for (const auto& elem : results) {
    std::cout << elem << " ";
  }
  std::cout << std::endl;
}