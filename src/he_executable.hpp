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

#pragma once

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "he_backend.hpp"
#include "he_tensor.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/util.hpp"
#include "node_wrapper.hpp"
#include "seal/seal.h"

namespace ngraph {
namespace runtime {
namespace he {

class HEExecutable : public Executable {
 public:
  HEExecutable(const std::shared_ptr<Function>& function,
               bool enable_performance_collection,
               const runtime::he::HEBackend* he_backend, bool encrypt_data,
               bool encrypt_model, bool batch_data);

  ~HEExecutable() {
    NGRAPH_INFO << "~HEExecutable()";
    /*NGRAPH_INFO << "Waiting until write completes";
    while (!m_is_writing) {

    }
    NGRAPH_INFO << "Write complete"; */
    // std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // std::cout << "Stopping session" << std::endl;
    // boost::asio::post(m_io_context, [this]() { m_session->socket().close();
    // });

    // TODO: cleaner way to prevent m_acceptor from double-freeing
    m_acceptor = nullptr;
    m_session = nullptr;
    // std::cout << "Stopped session" << std::endl;

    m_thread.join();
    NGRAPH_INFO << "done with ~HEExecutable() ";
  }

  /// @brief starts the server
  void start_server();

  bool call(const std::vector<std::shared_ptr<Tensor>>& outputs,
            const std::vector<std::shared_ptr<Tensor>>& inputs) override;

  void he_validate(
      const std::vector<std::shared_ptr<runtime::he::HETensor>>& outputs,
      const std::vector<std::shared_ptr<runtime::he::HETensor>>& inputs);

  std::vector<PerformanceCounter> get_performance_data() const override;

  size_t get_port() const { return m_port; };

  void accept_connection();

  void handle_message(const TCPMessage& message);

 private:
  bool m_encrypt_data;
  bool m_batch_data;
  bool m_encrypt_model;
  bool m_is_compiled = false;
  const HEBackend* m_he_backend = nullptr;  // TODO: replace with context
  std::unordered_map<const Node*, stopwatch> m_timer_map;
  std::vector<NodeWrapper> m_wrapped_nodes;

  std::shared_ptr<tcp::acceptor> m_acceptor;
  std::shared_ptr<TCPSession> m_session;
  std::shared_ptr<TCPServer> m_tcp_server;
  std::thread m_thread;
  boost::asio::io_context m_io_context;
  bool m_session_started{false};
  bool m_is_writing{false};
  std::vector<std::shared_ptr<runtime::he::HETensor>>
      m_inputs;  // (Encrypted) inputs to compiled function
  std::vector<std::shared_ptr<runtime::Tensor>>
      m_outputs;  // (Encrypted) outputs of compiled function

  size_t m_port{34000};  // Which port the server is hosted at

  std::shared_ptr<seal::SEALContext>
      m_context;  // TODO: move to he_seal_executable.hpp

  TCPMessage m_result_message;

  void generate_calls(const element::Type& type, const NodeWrapper& op,
                      const std::vector<std::shared_ptr<HETensor>>& outputs,
                      const std::vector<std::shared_ptr<HETensor>>& inputs);
};

}  // namespace he
}  // namespace runtime
}  // namespace ngraph
