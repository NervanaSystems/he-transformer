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
#include <set>
#include <thread>
#include <vector>

#include "he_op_annotations.hpp"
#include "he_tensor.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/util.hpp"
#include "node_wrapper.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "tcp/tcp_message.hpp"
#include "tcp/tcp_session.hpp"

using boost::asio::ip::tcp;

namespace ngraph {
namespace he {
/// \brief Class representing a function to execute
class HESealExecutable : public runtime::Executable {
 public:
  /// \brief Constructs an exectuable object
  /// \param[in] function Function in the executable
  /// \param[in] enable_performance_collection Unused: TODO use
  /// \param[in] he_seal_backend Backend storing encryption context
  /// \param[in] enable_client Whether or not to rely on a client to store the
  /// secret key
  HESealExecutable(const std::shared_ptr<Function>& function,
                   bool enable_performance_collection,
                   HESealBackend& he_seal_backend, bool enable_client);

  /// \brief Shuts down the TCP session if client is enabled
  ~HESealExecutable() override;

  /// \brief Prepares for inference on the function using a server
  /// \returns True if setup was successful, false otherwise
  bool server_setup();

  /// \brief Starts the server, which awaits a connection from a client
  void start_server();

  void update_he_op_annotations();

  /// \brief Calls the executable on the given input tensors.
  /// If the client is enabled, the inputs are dummy values and ignored.
  /// Instead, the inputs will be provided by the client
  /// \param[in] inputs Input tensor arguments to the function.
  /// \param[out] outputs Output tensors storing the result of the
  /// function
  bool call(
      const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
      const std::vector<std::shared_ptr<runtime::Tensor>>& inputs) override;

  // TOOD
  std::vector<runtime::PerformanceCounter> get_performance_data()
      const override;

  // \brief Returns the port at which the server is expecting a connection
  size_t get_port() const { return m_port; }

  // TODO: merge _done() methods

  /// \brief Returns whether or not the maxpool op has completed
  bool max_pool_done() const { return m_max_pool_done; }

  /// \brief Returns whether or not the minimum op has completed
  bool minimum_done() const { return m_minimum_done; }

  /// \brief Returns whether or not the session has started
  bool session_started() const { return m_session_started; }

  /// \brief Returns whether or not the client has provided input data to call
  /// the function
  bool client_inputs_received() const { return m_client_inputs_received; }

  void accept_connection();

  /// \brief Returns whether or not encryption parameters use complex packing
  inline bool complex_packing() const {
    return m_he_seal_backend.get_encryption_parameters().complex_packing();
  }

  /// \brief Checks whether or not the client supports the function
  /// \throws ngraph_error if function is unsupported
  /// Currently, we only support functions with a single client parameter and
  /// single results
  void check_client_supports_function();

  /// \brief Processes a message from the client
  /// \param[in] message Message to process
  void handle_message(const TCPMessage& message);

  /// \brief Processes a client message with ciphertexts to call the function
  /// \param[in] proto_msg Message to process
  void handle_client_ciphers(const he_proto::TCPMessage& proto_msg);

  /// \brief Processes a client message with ciphertextss after a ReLU function
  /// \param[in] proto_msg Message to process
  void handle_relu_result(const he_proto::TCPMessage& proto_msg);

  /// \brief Processes a client message with ciphertextss after a BoundedReLU
  /// function
  /// \param[in] proto_msg Message to process
  void handle_bounded_relu_result(const he_proto::TCPMessage& proto_msg);

  /// \brief Processes a client message with ciphertextss after a MaxPool
  /// function
  /// \param[in] proto_msg Message to process
  void handle_max_pool_result(const he_proto::TCPMessage& proto_msg);

  /// \brief Sends results to the client
  void send_client_results();

  /// \brief Sends function's parameter shape to the client
  void send_inference_shape();

  /// \brief Loads the public key from the message
  /// \param[in] proto_msg from which to load the public key
  void load_public_key(const he_proto::TCPMessage& proto_msg);

  /// \brief Loads the evaluation key from the message
  /// \param[in] proto_msg from which to load the evluation key
  void load_eval_key(const he_proto::TCPMessage& proto_msg);

  /// \brief Processes the ReLU operation if the client is enabled
  /// \param[in] arg0_cipher Encrypted tensor argumnet
  /// \param[out] out_cipher Encrypted tensor result
  /// \param[in] node_wrapper Wrapper around operation to perform
  // TODO: rename
  void handle_server_relu_op(std::shared_ptr<HESealCipherTensor>& arg0_cipher,
                             std::shared_ptr<HESealCipherTensor>& out_cipher,
                             const NodeWrapper& node_wrapper);

  /// \brief Processes the MaxPool operation if the client is enabled
  /// \param[in] arg0_cipher Encrypted tensor argumnet
  /// \param[out] out_cipher Encrypted tensor result
  /// \param[in] node_wrapper Wrapper around operation to perform
  // TODO: rename
  void handle_server_max_pool_op(
      std::shared_ptr<HESealCipherTensor>& arg0_cipher,
      std::shared_ptr<HESealCipherTensor>& out_cipher,
      const NodeWrapper& node_wrapper);

  /// \brief Returns whether or not a node's verbosity is on or off
  /// \param[in] op Operation to determine verbosity of
  inline bool verbose_op(const ngraph::Node& op) {
    return m_verbose_all_ops ||
           m_verbose_ops.find(ngraph::to_lower(op.description())) !=
               m_verbose_ops.end();
  }

  /// \brief Returns whether or not a node dessccription verbosity is on or off
  /// \param[in] description Node description determine verbosity of
  inline bool verbose_op(const std::string& description) {
    return m_verbose_all_ops ||
           m_verbose_ops.find(ngraph::to_lower(description)) !=
               m_verbose_ops.end();
  }

  /// \brief Returns the batch size
  inline size_t batch_size() const { return m_batch_size; }

  /// \brief Returns the batch size
  void set_batch_size(size_t batch_size);

  /// \brief Returns whether or not operation node should be received from
  /// client. Defaults to false if op has no HEOpAnnotation.
  /// \param[in] op Graph operation, should be Constant or Parameter node
  inline bool from_client(const ngraph::op::Op& op) {
    auto annotation = op.get_op_annotations();
    if (auto he_annotation =
            std::dynamic_pointer_cast<HEOpAnnotations>(annotation)) {
      NGRAPH_HE_LOG(5) << "Op has he annotation " << *he_annotation;
      return he_annotation->from_client();
    }
    NGRAPH_HE_LOG(5) << "Op has no he annotation";
    return false;
  }

  /// \brief Returns whether or not operation node should be packed using
  /// plaintext packing. Defaults to false if op has no HEOpAnnotation.
  /// \param[in] node_wrapper Wrapper of graph operation
  inline bool plaintext_packed(const NodeWrapper& node_wrapper) {
    return plaintext_packed(*node_wrapper.get_op());
  }

  /// \brief Returns whether or not operation node should be packed using
  /// plaintext packing. Defaults to false if op has no HEOpAnnotation.
  /// \param[in] op Graph operation
  inline bool plaintext_packed(const ngraph::op::Op& op) {
    auto annotation = op.get_op_annotations();
    if (auto he_annotation =
            std::dynamic_pointer_cast<HEOpAnnotations>(annotation)) {
      return he_annotation->packed();
    }
    return false;
  }

 private:
  HESealBackend& m_he_seal_backend;
  bool m_is_compiled;
  bool m_verbose_all_ops;
  std::shared_ptr<Function> m_function;

  bool m_sent_inference_shape{false};
  bool m_client_public_key_set{false};
  bool m_client_eval_key_set{false};

  bool m_enable_client;
  bool m_server_setup;
  size_t m_batch_size;
  size_t m_port;  // Which port the server is hosted at

  std::unordered_map<std::shared_ptr<const Node>, stopwatch> m_timer_map;
  std::vector<NodeWrapper> m_wrapped_nodes;

  std::unique_ptr<tcp::acceptor> m_acceptor;

  // Must be shared, since TCPSession uses enable_shared_from_this()
  std::shared_ptr<TCPSession> m_session;
  std::thread m_message_handling_thread;
  boost::asio::io_context m_io_context;

  // (Encrypted) inputs to compiled function
  std::vector<std::shared_ptr<HETensor>> m_client_inputs;
  std::vector<size_t> m_client_load_idx;
  // (Encrypted) outputs of compiled function
  std::vector<std::shared_ptr<HETensor>> m_client_outputs;

  std::vector<std::shared_ptr<SealCiphertextWrapper>> m_relu_ciphertexts;
  std::vector<std::shared_ptr<SealCiphertextWrapper>> m_max_pool_ciphertexts;
  std::vector<std::shared_ptr<SealCiphertextWrapper>> m_minimum_ciphertexts;

  std::set<std::string> m_verbose_ops;

  std::shared_ptr<seal::SEALContext> m_context;

  // To trigger when relu is done
  std::mutex m_relu_mutex;
  std::condition_variable m_relu_cond;
  size_t m_relu_done_count;
  std::vector<size_t> m_unknown_relu_idx;

  // To trigger when max_pool is done
  std::mutex m_max_pool_mutex;
  std::condition_variable m_max_pool_cond;
  bool m_max_pool_done;

  // To trigger when minimum is done
  std::mutex m_minimum_mutex;
  std::condition_variable m_minimum_cond;
  bool m_minimum_done;

  // To trigger when result message has been written
  std::mutex m_result_mutex;
  std::condition_variable m_result_cond;

  // To trigger when session has started
  std::mutex m_session_mutex;
  std::condition_variable m_session_cond;
  bool m_session_started;

  // To trigger when client inputs have been received
  std::mutex m_client_inputs_mutex;
  std::condition_variable m_client_inputs_cond;
  bool m_client_inputs_received;

  void generate_calls(const element::Type& type, const NodeWrapper& op,
                      const std::vector<std::shared_ptr<HETensor>>& outputs,
                      const std::vector<std::shared_ptr<HETensor>>& inputs);

  bool m_stop_const_fold{flag_to_bool(std::getenv("STOP_CONST_FOLD"))};
};
}  // namespace he
}  // namespace ngraph
