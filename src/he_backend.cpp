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

#include <chrono>
#include <limits>
#include <thread>

#include "he_backend.hpp"
#include "he_cipher_tensor.hpp"
#include "he_executable.hpp"
#include "he_plain_tensor.hpp"
#include "he_tensor.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/function.hpp"

using namespace ngraph;
using namespace std;

using descriptor::layout::DenseTensorLayout;

void runtime::he::HEBackend::start_server() {
  // Server
  tcp::resolver resolver(m_io_context);
  tcp::endpoint server_endpoints(tcp::v4(), m_port);

  m_acceptor = make_shared<tcp::acceptor>(m_io_context, server_endpoints);

  accept_connection();

  // m_tcp_server =
  //    make_shared<TCPServer>(m_io_context, server_endpoints, server_callback);
  m_thread = std::thread([this]() { m_io_context.run(); });
  // m_thread =
  // m_io_context.run();  // Actually start the server
}

void runtime::he::HEBackend::accept_connection() {
  // std::lock_guard<std::mutex> guard(m_session_mutex);
  std::cout << "Server accepting connections" << std::endl;

  auto server_callback = std::bind(&runtime::he::HEBackend::handle_message,
                                   this, std::placeholders::_1);

  /*auto server_callback = [this](const runtime::he::TCPMessage& message) {
    this->handle_message(message);
  };*/

  m_acceptor->async_accept([this, server_callback](boost::system::error_code ec,
                                                   tcp::socket socket) {
    if (!ec) {
      std::cout << "Connection accepted" << std::endl;
      // TODO: use make_shared here without causing seg-fault
      m_session =
          std::make_unique<TCPSession>(std::move(socket), server_callback);
      m_session->start();

      m_session_started = true;  // TODO: cleaner way to process this
      std::cout << "TCP session started" << std::endl;
    } else {
      std::cout << "error " << ec.message() << std::endl;
    }
    // accept_connection();
  });
}

shared_ptr<runtime::he::HEPlaintext>
runtime::he::HEBackend::create_valued_plaintext(
    float value, const element::Type& element_type) const {
  const string type_name = element_type.c_type_string();
  shared_ptr<runtime::he::HEPlaintext> plaintext = create_empty_plaintext();

  encode(plaintext, (void*)(&value), element_type, 1);
  return plaintext;
}

shared_ptr<runtime::he::HECiphertext>
runtime::he::HEBackend::create_valued_ciphertext(
    float value, const element::Type& element_type, size_t batch_size) const {
  if (batch_size != 1) {
    throw ngraph_error(
        "HEBackend::create_valued_ciphertext only supports batch size 1");
  }
  const string type_name = element_type.c_type_string();
  shared_ptr<runtime::he::HEPlaintext> plaintext =
      create_valued_plaintext(value, element_type);
  shared_ptr<runtime::he::HECiphertext> ciphertext = create_empty_ciphertext();

  encrypt(ciphertext, *plaintext);
  return ciphertext;
}

shared_ptr<runtime::Tensor> runtime::he::HEBackend::create_tensor(
    const element::Type& element_type, const Shape& shape,
    void* memory_pointer) {
  throw ngraph_error("HE create_tensor unimplemented");
}

shared_ptr<runtime::Tensor> runtime::he::HEBackend::create_tensor(
    const element::Type& element_type, const Shape& shape) {
  if (batch_data()) {
    return create_batched_cipher_tensor(element_type, shape);
  } else {
    return create_cipher_tensor(element_type, shape);
  }
}

shared_ptr<runtime::Tensor> runtime::he::HEBackend::create_plain_tensor(
    const element::Type& element_type, const Shape& shape,
    const bool batched) const {
  auto rc = make_shared<runtime::he::HEPlainTensor>(
      element_type, shape, this, create_empty_plaintext(), batched);
  return static_pointer_cast<runtime::Tensor>(rc);
}

shared_ptr<runtime::Tensor> runtime::he::HEBackend::create_cipher_tensor(
    const element::Type& element_type, const Shape& shape,
    const bool batched) const {
  auto rc = make_shared<runtime::he::HECipherTensor>(
      element_type, shape, this, create_empty_ciphertext(), batched);
  return static_pointer_cast<runtime::Tensor>(rc);
}

shared_ptr<runtime::Tensor> runtime::he::HEBackend::create_valued_cipher_tensor(
    float value, const element::Type& element_type, const Shape& shape) const {
  auto tensor = static_pointer_cast<HECipherTensor>(
      create_cipher_tensor(element_type, shape));
  vector<shared_ptr<runtime::he::HECiphertext>>& cipher_texts =
      tensor->get_elements();
#pragma omp parallel for
  for (size_t i = 0; i < cipher_texts.size(); ++i) {
    cipher_texts[i] = create_valued_ciphertext(value, element_type);
  }
  return tensor;
}

shared_ptr<runtime::Tensor> runtime::he::HEBackend::create_valued_plain_tensor(
    float value, const element::Type& element_type, const Shape& shape) const {
  auto tensor = static_pointer_cast<HEPlainTensor>(
      create_plain_tensor(element_type, shape));
  vector<shared_ptr<runtime::he::HEPlaintext>>& plain_texts =
      tensor->get_elements();
#pragma omp parallel for
  for (size_t i = 0; i < plain_texts.size(); ++i) {
    plain_texts[i] = create_valued_plaintext(value, element_type);
  }
  return tensor;
}

std::shared_ptr<runtime::Executable> runtime::he::HEBackend::compile(
    shared_ptr<Function> function, bool enable_performance_collection) {
  return make_shared<HEExecutable>(function, enable_performance_collection,
                                   this, m_encrypt_data, m_encrypt_model,
                                   m_batch_data);
}
