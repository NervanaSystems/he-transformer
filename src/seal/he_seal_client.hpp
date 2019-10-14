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

#include <boost/asio.hpp>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "seal/he_seal_encryption_parameters.hpp"
#include "seal/seal.h"
#include "tcp/tcp_client.hpp"
#include "tcp/tcp_message.hpp"
#include "util.hpp"

namespace ngraph {
namespace he {

template <class T>
using HETensorConfigMap =
    std::unordered_map<std::string, std::pair<std::string, std::vector<T>>>;

/// \brief Class representing a data owner. The client provides encrypted values
/// to a server and receives the encrypted result. The client may also aid in
/// the computation, for example by computing activation functions the sever
/// cannot compute using homomorphic encryption
class HESealClient {
 public:
  /// \brief Constructs a client object and connects to a server
  /// \param[in] hostname Hostname of the server
  /// \param[in] port Port of the server
  /// \param[in] batch_size Batch size of the inference to perform
  /// \param[in] inputs Input data as a map from tensor name to pair of
  /// ('encrypt', inputs) or ('plain', inputs)
  HESealClient(const std::string& hostname, const size_t port,
               const size_t batch_size,
               const HETensorConfigMap<double>& inputs);

  /// \brief Constructs a client object and connects to a server
  /// \param[in] hostname Hostname of the server
  /// \param[in] port Port of the server
  /// \param[in] batch_size Batch size of the inference to perform
  /// \param[in] inputs Input data as a map from tensor name to inputs
  HESealClient(const std::string& hostname, const size_t port,
               const size_t batch_size, const HETensorConfigMap<float>& inputs);

  /// \brief Constructs a client object and connects to a server
  /// \param[in] hostname Hostname of the server
  /// \param[in] port Port of the server
  /// \param[in] batch_size Batch size of the inference to perform
  /// \param[in] inputs Input data as a map from tensor name to inputs
  HESealClient(const std::string& hostname, const size_t port,
               const size_t batch_size,
               const HETensorConfigMap<int64_t>& inputs);

  /// \brief Creates SEAL context
  void set_seal_context();

  /// \brief Processes a message from the server
  /// \param[in] message Message to process
  void handle_message(const ngraph::he::TCPMessage& message);

  /// \brief Processes a message containing encryption parameters
  /// \param[in] message Message to process
  void handle_encryption_parameters_response(
      const he_proto::TCPMessage& message);

  /// \brief Processes a request to perform ReLU function
  /// \param[in] message Message to process
  void handle_relu_request(he_proto::TCPMessage&& message);

  /// \brief Processes a request to perform MaxPool function
  /// \param[in] message Message to process
  void handle_max_pool_request(he_proto::TCPMessage&& message);

  /// \brief Processes a request to perform BoundedReLU function
  /// \param[in] message Message to process
  void handle_bounded_relu_request(he_proto::TCPMessage&& message);

  /// \brief Processes a message containing the result from the server
  /// \param[in] message Message to process
  void handle_result(const he_proto::TCPMessage& message);

  /// \brief Processes a message containing the inference shape
  /// \param[in] message Message to process
  void handle_inference_request(const he_proto::TCPMessage& message);

  /// \brief Sends the public key and relinearization keys to the server
  void send_public_and_relin_keys();

  /// \brief Writes a mesage to the server
  /// \param[in] message Message to write
  inline void write_message(const ngraph::he::TCPMessage&& message) {
    m_tcp_client->write_message(std::move(message));
  }

  /// \brief Returns whether or not the function is done evaluating
  inline bool is_done() { return m_is_done; }

  /// \brief Returns decrypted results
  /// \warning Will lock until results are ready
  inline std::vector<double> get_results() {
    NGRAPH_HE_LOG(1) << "Client waiting for results";

    std::unique_lock<std::mutex> mlock(m_is_done_mutex);
    m_is_done_cond.wait(mlock, std::bind(&HESealClient::is_done, this));
    return m_results;
  }

  /// \brief Closes conection with the server
  void close_connection();

  /// \brief Returns whether or not the encryption parameters use complex
  /// packing
  inline bool complex_packing() const {
    return m_encryption_params.complex_packing();
  }

  /// \brief Returns the scale of the encryption parameters
  inline double scale() const { return m_encryption_params.scale(); }

 private:
  std::unique_ptr<TCPClient> m_tcp_client;
  ngraph::he::HESealEncryptionParameters m_encryption_params;
  std::shared_ptr<seal::PublicKey> m_public_key;
  std::shared_ptr<seal::SecretKey> m_secret_key;
  std::shared_ptr<seal::SEALContext> m_context;
  std::shared_ptr<seal::Encryptor> m_encryptor;
  std::shared_ptr<seal::CKKSEncoder> m_ckks_encoder;
  std::shared_ptr<seal::Decryptor> m_decryptor;
  std::shared_ptr<seal::Evaluator> m_evaluator;
  std::shared_ptr<seal::KeyGenerator> m_keygen;
  std::shared_ptr<seal::RelinKeys> m_relin_keys;
  size_t m_batch_size;
  bool m_is_done;
  std::condition_variable m_is_done_cond;
  std::mutex m_is_done_mutex;

  // Function inputs and configuration
  HETensorConfigMap<double> m_input_config;
  std::vector<double> m_results;  // Function outputs

};  // namespace he
}  // namespace he
}  // namespace ngraph
