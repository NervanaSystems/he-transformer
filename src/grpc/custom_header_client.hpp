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
