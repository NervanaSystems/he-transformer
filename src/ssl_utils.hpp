//*****************************************************************************
// Copyright 2018 Intel Corporation
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

#include <arpa/inet.h>
#include <openssl/err.h>
#include <openssl/ssl.h>
#include <stdio.h>
#include <sys/socket.h>
#include <unistd.h>

namespace ngraph {
namespace runtime {
namespace he {
namespace ssl {

int create_socket(int port);

void init_openssl();

void cleanup_openssl();

SSL_CTX* create_context();

void configure_context(SL_CTX* ctx);

}  // namespace ssl
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
