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
#include <memory>

#include "boost/asio.hpp"
#include "gtest/gtest.h"
#include "logging/ngraph_he_log.hpp"
#include "protos/message.pb.h"
#include "seal/he_seal_backend.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "tcp/tcp_client.hpp"
#include "tcp/tcp_message.hpp"
#include "tcp/tcp_session.hpp"
#include "util/test_tools.hpp"

namespace ngraph::runtime::he {

auto dummy_tcp_message = []() {
  // Write message
  pb::TCPMessage proto_msg;
  pb::Function f;
  f.set_function("123");
  *proto_msg.mutable_function() = f;
  std::stringstream s;
  proto_msg.SerializeToOstream(&s);
  TCPMessage tcp_message(std::move(proto_msg));

  return tcp_message;
};

class MockServer {
 public:
  MockServer(size_t port, size_t message_cnt) {
    boost::asio::ip::tcp::resolver resolver(m_io_context);
    boost::asio::ip::tcp::endpoint server_endpoints(boost::asio::ip::tcp::v4(),
                                                    port);
    m_acceptor = std::make_unique<boost::asio::ip::tcp::acceptor>(
        m_io_context, server_endpoints);
    boost::asio::socket_base::reuse_address option(true);
    m_acceptor->set_option(option);

    auto server_callback = [](const TCPMessage& message) { return; };

    NGRAPH_HE_LOG(1) << "Server calling async_accept";
    m_acceptor->async_accept([this, server_callback](
                                 boost::system::error_code ec,
                                 boost::asio::ip::tcp::socket socket) {
      if (!ec) {
        m_session =
            std::make_shared<TCPSession>(std::move(socket), server_callback);
        m_session->start();

        std::lock_guard<std::mutex> guard(m_session_mutex);
        m_session_started = true;
        m_session_cond.notify_one();

      } else {
        NGRAPH_ERR << "error accepting connection " << ec.message();
      }
    });

    m_message_handling_thread = std::thread([this]() {
      try {
        m_io_context.run();
      } catch (std::exception& e) {
        NGRAPH_CHECK(false, "Server error handling thread: ", e.what());
      }
    });

    std::unique_lock<std::mutex> mlock(m_session_mutex);
    NGRAPH_HE_LOG(3) << "waiting thread got mutex";
    m_session_cond.wait(mlock, [this]() { return m_session_started; });

    for (size_t i = 0; i < message_cnt; ++i) {
      m_session->write_message(dummy_tcp_message());
    }
  }

  ~MockServer() {
    if (m_message_handling_thread.joinable()) {
      NGRAPH_HE_LOG(5) << "Waiting for m_message_handling_thread to join";
      try {
        m_message_handling_thread.join();
      } catch (std::exception& e) {
        NGRAPH_ERR << "Exception closing executable thread " << e.what();
      }
      NGRAPH_HE_LOG(5) << "m_message_handling_thread joined";
    }

    // m_acceptor and m_io_context both free the socket? Avoid double-free
    try {
      m_acceptor->close();
    } catch (std::exception& e) {
      NGRAPH_ERR << "Exception closing m_acceptor " << e.what();
    }
    m_acceptor = nullptr;
    m_session = nullptr;
  }

 private:
  std::unique_ptr<boost::asio::ip::tcp::acceptor> m_acceptor;
  std::shared_ptr<TCPSession> m_session;
  std::thread m_message_handling_thread;
  boost::asio::io_context m_io_context;
  std::mutex m_session_mutex;

  std::condition_variable m_session_cond;
  bool m_session_started{false};
};

class MockClient {
 public:
  MockClient(std::string hostname, size_t port, size_t max_message_count)
      : m_max_message_count{max_message_count} {
    boost::asio::io_context io_context;
    boost::asio::ip::tcp::resolver resolver(io_context);
    auto endpoints = resolver.resolve(hostname, std::to_string(port));
    auto client_callback = [this](const TCPMessage& message) {
      m_message_count++;

      if (m_message_count < m_max_message_count) {
        for (size_t i = 0; i < m_max_message_count; ++i) {
          TCPMessage return_message(message);
          m_tcp_client->write_message(std::move(return_message));
        }
      } else {
        m_tcp_client->close();
      }
    };
    m_tcp_client =
        std::make_unique<TCPClient>(io_context, endpoints, client_callback);
    io_context.run();
  }

 private:
  std::unique_ptr<TCPClient> m_tcp_client;

  size_t m_message_count{0};
  size_t m_max_message_count;
};

TEST(tcp_client, connect_before_server_started) {
  size_t port{34000};
  std::string hostname{"localhost"};
  size_t message_count{1};

  auto client_thread = std::thread(
      [&]() { auto client = MockClient(hostname, port, message_count); });

  // TODO(fboemer): Better way of guaranteeing client has started
  // Delay to check what happends when server starts after client
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  auto server = MockServer(port, message_count);
  client_thread.join();
}

// TOD(fboemer): Better way to check message queue is non-empty?
TEST(tcp_client, non_empty_message_queue) {
  size_t port{34000};
  std::string hostname{"localhost"};

  size_t message_count{100};

  auto client_thread = std::thread(
      [&]() { auto client = MockClient(hostname, port, message_count); });

  auto server = MockServer(port, message_count);
  client_thread.join();
}

}  // namespace ngraph::runtime::he
