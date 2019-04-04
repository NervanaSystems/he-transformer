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

#include <atomic>
#include <boost/asio.hpp>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "he_backend.hpp"
#include "he_tensor.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/util.hpp"
#include "node_wrapper.hpp"
#include "seal/seal.h"
#include "tcp/tcp_message.hpp"
#include "tcp/tcp_session.hpp"

using boost::asio::ip::tcp;

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
    if (m_enable_client) {
      // TODO: why is this needed to prevent m_acceptor from double-freeing?
      m_acceptor = nullptr;
      m_thread.join();
    }
  }

  void start_server();

  bool call(const std::vector<std::shared_ptr<Tensor>>& outputs,
            const std::vector<std::shared_ptr<Tensor>>& inputs) override;

  void he_validate(
      const std::vector<std::shared_ptr<runtime::he::HETensor>>& outputs,
      const std::vector<std::shared_ptr<runtime::he::HETensor>>& inputs);

  std::vector<PerformanceCounter> get_performance_data() const override;

  size_t get_port() const { return m_port; };

  // TODO: merge two _done() methods
  bool relu_done() const { return m_relu_done; };
  bool sort_done() const { return m_sort_done; };

  bool session_started() const { return m_session_started; };

  bool client_inputs_received() const { return m_client_inputs_received; };

  void accept_connection();

  void handle_message(const TCPMessage& message);

 private:
  const HEBackend* m_he_backend;  // TODO: replace with context
  bool m_encrypt_data;
  bool m_encrypt_model;
  bool m_batch_data;
  bool m_is_compiled;

  bool m_enable_client;
  size_t m_batch_size;
  size_t m_port;  // Which port the server is hosted at

  std::unordered_map<const Node*, stopwatch> m_timer_map;
  std::vector<NodeWrapper> m_wrapped_nodes;

  std::shared_ptr<tcp::acceptor> m_acceptor;
  std::shared_ptr<TCPSession> m_session;
  std::thread m_thread;
  boost::asio::io_context m_io_context;

  // (Encrypted) inputs to compiled function
  std::vector<std::shared_ptr<runtime::he::HETensor>> m_client_inputs;
  // (Encrypted) outputs of compiled function
  std::vector<std::shared_ptr<runtime::he::HETensor>> m_client_outputs;

  std::vector<std::shared_ptr<runtime::he::HECiphertext>> m_relu_ciphertexts;
  std::vector<std::shared_ptr<runtime::he::HECiphertext>> m_sort_ciphertexts;

  std::shared_ptr<seal::SEALContext>
      m_context;  // TODO: move to he_seal_executable.hpp

  // To trigger when relu is done
  std::mutex m_relu_mutex;
  std::condition_variable m_relu_cond;
  bool m_relu_done;

  // To trigger when sorting is done
  std::mutex m_sort_mutex;
  std::condition_variable m_sort_cond;
  bool m_sort_done;

  // To trigger when session has started
  std::mutex m_session_mutex;
  std::condition_variable m_session_cond;
  bool m_session_started;

  // To trigger when client inputs have been received
  std::mutex m_client_inputs_mutex;
  std::condition_variable m_client_inputs_cond;
  bool m_client_inputs_received;

  TCPMessage m_result_message;

  void generate_calls(const element::Type& type, const NodeWrapper& op,
                      const std::vector<std::shared_ptr<HETensor>>& outputs,
                      const std::vector<std::shared_ptr<HETensor>>& inputs);
};
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
