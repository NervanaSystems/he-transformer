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
#include <string>
#include <vector>

#include "seal/seal.h"
#include "tcp/tcp_client.hpp"
#include "tcp/tcp_message.hpp"
#include "util.hpp"

namespace ngraph {
namespace he {
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
  /// \param[in] inputs Input data
  /// \param[in] complex_packing Whether or not to use complex packing
  HESealClient(const std::string& hostname, const size_t port,
               const size_t batch_size, const std::vector<double>& inputs);

  /// \brief Constructs a client object and connects to a server
  /// \param[in] hostname Hostname of the server
  /// \param[in] port Port of the server
  /// \param[in] batch_size Batch size of the inference to perform
  /// \param[in] inputs Input data
  /// \param[in] complex_packing Whether or not to use complex packing
  HESealClient(const std::string& hostname, const size_t port,
               const size_t batch_size, const std::vector<float>& inputs);

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
  void handle_relu_request(const ngraph::he::TCPMessage& message);

  /// \brief TODO
  void handle_relu_request(he_proto::TCPMessage&& message);

  /// \brief Processes a request to perform MaxPool function
  /// \param[in] message Message to process
  void handle_max_pool_request(const he_proto::TCPMessage& message);

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

  /// \brief Returns whether or not the function has completed evaluation
  inline bool is_done() { return m_is_done; }

  /// \brief Returns decrypted results
  std::vector<double> get_results() { return m_results; }

  /// \brief Closes conection with the server
  void close_connection();

  /// \brief Returns whether or not complex packing is used
  bool complex_packing() const { return m_complex_packing; }

  /// \brief Returns whether or not complex packing is used
  bool& complex_packing() { return m_complex_packing; }

 private:
  std::unique_ptr<TCPClient> m_tcp_client;
  HESealEncryptionParameters m_encryption_params;
  std::shared_ptr<seal::PublicKey> m_public_key;
  std::shared_ptr<seal::SecretKey> m_secret_key;
  std::shared_ptr<seal::SEALContext> m_context;
  std::shared_ptr<seal::Encryptor> m_encryptor;
  std::shared_ptr<seal::CKKSEncoder> m_ckks_encoder;
  std::shared_ptr<seal::Decryptor> m_decryptor;
  std::shared_ptr<seal::Evaluator> m_evaluator;
  std::shared_ptr<seal::KeyGenerator> m_keygen;
  std::shared_ptr<seal::RelinKeys> m_relin_keys;
  double m_scale{0.0};
  size_t m_batch_size;
  bool m_is_done;
  std::vector<double> m_inputs;   // Function inputs
  std::vector<double> m_results;  // Function outputs

  bool m_complex_packing{false};
};  // namespace he
}  // namespace he
}  // namespace ngraph
