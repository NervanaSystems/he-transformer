#pragma once

#include <grpcpp/grpcpp.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include "helloworld.grpc.pb.h"

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