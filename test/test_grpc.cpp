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

#include <grpcpp/grpcpp.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include "helloworld.grpc.pb.h"

// TODO: better soluton
#include "helloworld.grpc.pb.cc"
#include "helloworld.pb.cc"

#include "gtest/gtest.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using helloworld::Greeter;
using helloworld::HelloReply;
using helloworld::HelloRequest;

TEST(grpc, trivial) { EXPECT_EQ(1, 1); }

TEST(grpc, greeter_client) {
  class CustomHeaderClient {
   public:
    CustomHeaderClient(std::shared_ptr<Channel> channel)
        : stub_(Greeter::NewStub(channel)) {}

    // Assembles the client's payload, sends it and presents the response back
    // from the server.
    std::string SayHello(const std::string& user) {
      // Data we are sending to the server.
      HelloRequest request;
      request.set_name(user);

      // Container for the data we expect from the server.
      HelloReply reply;

      // Context for the client. It could be used to convey extra information to
      // the server and/or tweak certain RPC behaviors.
      ClientContext context;

      // Setting custom metadata to be sent to the server
      context.AddMetadata("custom-header", "Custom Value");

      // Setting custom binary metadata
      char bytes[8] = {'\0', '\1', '\2', '\3', '\4', '\5', '\6', '\7'};
      context.AddMetadata("custom-bin", grpc::string(bytes, 8));

      // The actual RPC.
      Status status = stub_->SayHello(&context, request, &reply);

      // Act upon its status.
      if (status.ok()) {
        std::cout << "Client received initial metadata from server: "
                  << context.GetServerInitialMetadata()
                         .find("custom-server-metadata")
                         ->second
                  << std::endl;
        std::cout << "Client received trailing metadata from server: "
                  << context.GetServerTrailingMetadata()
                         .find("custom-trailing-metadata")
                         ->second
                  << std::endl;
        return reply.message();
      } else {
        std::cout << status.error_code() << ": " << status.error_message()
                  << std::endl;
        return "RPC failed";
      }
    }

   private:
    std::unique_ptr<Greeter::Stub> stub_;
  };

  // Logic and data behind the server's behavior.
  class GreeterServiceImpl final : public Greeter::Service {
    Status SayHello(ServerContext* context, const HelloRequest* request,
                    HelloReply* reply) override {
      std::string prefix("Hello ");

      // Get the client's initial metadata
      std::cout << "Client metadata: " << std::endl;
      const std::multimap<grpc::string_ref, grpc::string_ref> metadata =
          context->client_metadata();
      for (auto iter = metadata.begin(); iter != metadata.end(); ++iter) {
        std::cout << "Header key: " << iter->first << ", value: ";
        // Check for binary value
        size_t isbin = iter->first.find("-bin");
        if ((isbin != std::string::npos) && (isbin + 4 == iter->first.size())) {
          std::cout << std::hex;
          for (auto c : iter->second) {
            std::cout << static_cast<unsigned int>(c);
          }
          std::cout << std::dec;
        } else {
          std::cout << iter->second;
        }
        std::cout << std::endl;
      }

      context->AddInitialMetadata("custom-server-metadata",
                                  "initial metadata value");
      context->AddTrailingMetadata("custom-trailing-metadata",
                                   "trailing metadata value");
      reply->set_message(prefix + request->name());
      return Status::OK;
    }
  };

  auto RunServer = []() {
    std::string server_address("0.0.0.0:50051");
    GreeterServiceImpl service;

    ServerBuilder builder;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    // Register "service" as the instance through which we'll communicate
    // with clients. In this case it corresponds to an *synchronous*
    // service.
    builder.RegisterService(&service);
    // Finally assemble the server.
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << server_address << std::endl;

    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    // Wait for the server to shutdown. Note that some other thread must be
    // responsible for shutting down the server for this call to ever
    // return.
    server->Shutdown();
  };

  // Server code
  std::thread server_thread(RunServer);

  // Client code
  CustomHeaderClient greeter(grpc::CreateChannel(
      "localhost:50051", grpc::InsecureChannelCredentials()));
  std::string user("world");
  std::string reply = greeter.SayHello(user);
  std::cout << "Client received message: " << reply << std::endl;

  grpc::Server Shutdown();
  server_thread.join();
}