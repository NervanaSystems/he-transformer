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

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "he_plaintext.hpp"
#include "he_tensor.hpp"
#include "he_type.hpp"
#include "he_util.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/descriptor/layout/tensor_layout.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/pass/like_replacement.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/performance_counter.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/util.hpp"
#include "node_wrapper.hpp"
#include "seal/he_seal_encryption_parameters.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/seal_plaintext_wrapper.hpp"

namespace ngraph {
namespace runtime {
class BackendConstructor;
}  // namespace runtime
namespace he {
class HEType;
class SealCiphertextWrapper;
/// \brief Class representing a backend using the CKKS homomorphic encryption
/// scheme.
class HESealBackend : public runtime::Backend {
 public:
  /// \brief Constructs a backend with default parameter choice
  HESealBackend();
  /// \brief Constructs a backend with the given encryption parameters
  /// \param[in] parms Encryption parameters
  explicit HESealBackend(HESealEncryptionParameters parms);

  /// \brief Prepares the backend with the encryption context, including
  /// generating encryption keys, encryptor, decryptor, evaluator, and encoder
  void generate_context();

  /// \brief Constructs an unpacked plaintext tensor
  /// \param[in] type Datatype to store in the tensor
  /// \param[in] shape Shape of the tensor
  std::shared_ptr<runtime::Tensor> create_tensor(const element::Type& type,
                                                 const Shape& shape) override;

  /// \brief Unimplemented
  std::shared_ptr<runtime::Tensor> create_tensor(
      const element::Type& /*type*/, const Shape& /*shape*/,
      void* /*memory_pointer*/) override {
    throw ngraph_error("create_tensor unimplemented");
  }

  /// \brief Compiles a function
  /// \brief param[in] function Function to compile
  /// \brief param[in] enable_performance_data TODO(fboemer): unused
  /// \returns An executable object
  std::shared_ptr<ngraph::runtime::Executable> compile(
      std::shared_ptr<Function> function,
      bool enable_performance_data = false) override;

  /// \brief Returns whether or not a given operation is supported
  /// \param[in] node Node representing an operation
  bool is_supported(const Node& node) const override;

  /// \brief Sets a configuration for the backend
  /// \param[in] config Configuration map. It should contain entries in one of
  /// the following forms:
  ///     1) {tensor_name : "client_input"}, which indicates the specified
  ///     tensor should be loaded from the client. Note, the tensor may or may
  ///     not be encrypted, as determined by the client.
  ///     2) {enable_client : "True" /"False"}, which indicates whether or not
  ///      the client should be enabled
  ///     3) {tensor_name : "encrypt"}, which indicates the specified
  ///     tensor should be encrypted. By default, tensors may or may not be
  ///     encrypted. Setting this option will encrypt the plaintext tensor of
  ///     name tensor_name if not already encrypted and it is not a client
  ///     input.
  ///     4) {tensor_name : "packed"}, which indicates the specified tensor
  ///     should use plaintext packing.
  ///     5) {"encryption_parameters" : "filename
  ///     or json string"}, which sets the encryption parameters to use.
  ///
  ///     Note, entries with the same tensor key should be comma-separated, for
  ///     instance: {tensor_name : "client_input,encrypt,packed"}
  ///
  ///  \warning Specfying entries of form 1) without an entry of form 2) will
  ///  not load the tensors from the client
  /// \param[out] error Error string. Unused
  bool set_config(const std::map<std::string, std::string>& config,
                  std::string& error) override;

  /// \brief Returns whether or not a given datatype is supported
  /// \param[in] type Datatype
  /// \returns True if datatype is supported, false otherwise
  bool is_supported_type(const ngraph::element::Type& type) const {
    return m_supported_types.find(type.hash()) != m_supported_types.end();
  }

  /// \brief Creates a cipher tensor using plaintext packing along the batch
  /// (i.e. first) axis
  /// \param[in] type Datatype stored in the tensor
  /// \param[in] shape Shape of the tensor
  /// \returns Pointer to created tensor
  std::shared_ptr<runtime::Tensor> create_packed_cipher_tensor(
      const element::Type& type, const Shape& shape) const;

  /// \brief Creates a plaintext tensor using plaintext packing along the batch
  /// (i.e. first) axis
  /// \param[in] type Datatype stored in the tensor
  /// \param[in] shape Shape of the tensor
  /// \returns Pointer to created tensor
  std::shared_ptr<runtime::Tensor> create_packed_plain_tensor(
      const element::Type& type, const Shape& shape) const;

  /// \brief Creates a plaintext tensor
  /// \param[in] type Datatype stored in the tensor
  /// \param[in] shape Shape of the tensor
  /// \param[in] plaintext_packing Whether or not to use plaintext packing
  /// \param[in] name Name of the created tensor
  /// \returns Pointer to created tensor
  std::shared_ptr<runtime::Tensor> create_plain_tensor(
      const element::Type& type, const Shape& shape,
      const bool plaintext_packing = false,
      const std::string& name = "external") const;

  /// \brief Creates a ciphertext tensor
  /// \param[in] type Datatype stored in the tensor
  /// \param[in] shape Shape of the tensor
  /// \param[in] plaintext_packing Whether or not to use plaintext packing
  /// \param[in] name Name of the created tensor
  /// \returns Pointer to created tensor
  std::shared_ptr<runtime::Tensor> create_cipher_tensor(
      const element::Type& type, const Shape& shape,
      const bool plaintext_packing = false,
      const std::string& name = "external") const;

  /// \brief Creates empty ciphertext
  /// \returns Pointer to created ciphertext
  static std::shared_ptr<SealCiphertextWrapper> create_empty_ciphertext() {
    return std::make_shared<SealCiphertextWrapper>();
  }

  /// \brief TODO(fboemer)
  void decode(void* output, const HEPlaintext& input, const element::Type& type,
              size_t count = 1) const;

  /// \brief TODO(fboemer)
  void encrypt(std::shared_ptr<SealCiphertextWrapper>& output,
               const HEPlaintext& input, const element::Type& type,
               bool complex_packing = false) const;

  /// \brief TODO(fboemer)
  void decrypt(HEPlaintext& output, const SealCiphertextWrapper& input,
               const bool complex_packing) const;

  /// \brief Returns pointer to SEAL context
  const std::shared_ptr<seal::SEALContext> get_context() const {
    return m_context;
  }

  /// \brief Returns pointer to relinearization keys
  const std::shared_ptr<seal::RelinKeys> get_relin_keys() const {
    return m_relin_keys;
  }

  /// \brief Returns pointer to Galois keys
  const std::shared_ptr<seal::GaloisKeys> get_galois_keys() const {
    return m_galois_keys;
  }

  /// \brief Returns pointer to encryptor
  const std::shared_ptr<seal::Encryptor> get_encryptor() const {
    return m_encryptor;
  }

  /// \brief Returns pointer to decryptor
  const std::shared_ptr<seal::Decryptor> get_decryptor() const {
    return m_decryptor;
  }

  /// \brief Retursn a pointer to evaluator
  const std::shared_ptr<seal::Evaluator> get_evaluator() const {
    return m_evaluator;
  }

  /// \brief Returns the encryption parameters
  const HESealEncryptionParameters& get_encryption_parameters() const {
    return m_encryption_params;
  }

  /// \brief Updates encryption parameters. Re-generates context and keys if
  /// necessary
  /// \param[in] new_parms New encryption parameters
  void update_encryption_parameters(
      const HESealEncryptionParameters& new_parms);

  /// \brief Returns the CKKS encoder
  const std::shared_ptr<seal::CKKSEncoder> get_ckks_encoder() const {
    return m_ckks_encoder;
  }

  /// \brief Sets the relinearization keys. Note, they may not be compatible
  /// with the other SEAL keys
  /// \param[in] keys relinearization keys
  void set_relin_keys(const seal::RelinKeys& keys) {
    m_relin_keys = std::make_shared<seal::RelinKeys>(keys);
  }

  /// \brief Sets the public keys. Note, they may not be compatible
  /// with the other SEAL keys
  /// \param[in] key public key
  void set_public_key(const seal::PublicKey& key) {
    m_public_key = std::make_shared<seal::PublicKey>(key);
    m_encryptor = std::make_shared<seal::Encryptor>(m_context, *m_public_key);
  }

  /// \brief TODO(fboemer)
  const std::unordered_map<std::uint64_t, std::uint64_t>& barrett64_ratio_map()
      const {
    return m_barrett64_ratio_map;
  }

  /// \brief Returns the top-level scale used for encoding
  double get_scale() const { return m_encryption_params.scale(); }

  /// \brief Returns whether or not complex packing is used
  bool complex_packing() const { return m_encryption_params.complex_packing(); }

  /// \brief Returns the chain index, also known as level, of the ciphertext
  /// \param[in] cipher Ciphertext whose chain index to return
  /// \returns The chain index of the ciphertext.
  size_t get_chain_index(const SealCiphertextWrapper& cipher) const {
    return m_context->get_context_data(cipher.ciphertext().parms_id())
        ->chain_index();
  }

  /// \brief Returns the chain index, also known as level, of the plaintext
  /// \param[in] plain Plaintext whose chain index to return
  /// \returns The chain index of the ciphertext.
  size_t get_chain_index(const SealPlaintextWrapper& plain) const {
    return m_context->get_context_data(plain.plaintext().parms_id())
        ->chain_index();
  }

  /// \brief Returns set of tensors to be provided by the client
  std::unordered_set<std::string> get_client_tensor_names() const {
    return m_client_tensor_names;
  }

  /// \brief Returns set of parameter tensors to be encrypted
  std::unordered_set<std::string> get_encrypted_tensor_names() const {
    return m_encrypted_tensor_names;
  }

  /// \brief Returns set of parameter tensors to remain plaintext.
  std::unordered_set<std::string> get_plaintext_tensor_names() const {
    return m_plaintext_tensor_names;
  }

  /// \brief Returns set of parameter tensors to be packed.
  std::unordered_set<std::string> get_packed_tensor_names() const {
    return m_packed_tensor_names;
  }

 private:
  bool m_enable_client{false};

  std::shared_ptr<seal::SecretKey> m_secret_key;
  std::shared_ptr<seal::PublicKey> m_public_key;
  std::shared_ptr<seal::RelinKeys> m_relin_keys;
  std::shared_ptr<seal::Encryptor> m_encryptor;
  std::shared_ptr<seal::Decryptor> m_decryptor;
  std::shared_ptr<seal::SEALContext> m_context;
  std::shared_ptr<seal::Evaluator> m_evaluator;
  std::shared_ptr<seal::KeyGenerator> m_keygen;
  std::shared_ptr<seal::GaloisKeys> m_galois_keys;
  HESealEncryptionParameters m_encryption_params;
  std::shared_ptr<seal::CKKSEncoder> m_ckks_encoder;

  // Stores Barrett64 ratios for moduli under 30 bits
  std::unordered_map<std::uint64_t, std::uint64_t> m_barrett64_ratio_map;

  std::unordered_set<size_t> m_supported_types{
      element::f32.hash(), element::i32.hash(), element::i64.hash(),
      element::f64.hash()};

  std::unordered_set<std::string> m_client_tensor_names;
  std::unordered_set<std::string> m_encrypted_tensor_names;
  std::unordered_set<std::string> m_plaintext_tensor_names;
  std::unordered_set<std::string> m_packed_tensor_names;

  std::unordered_set<std::string> m_unsupported_op_name_list{
      "Abs",
      "Acos",
      "All",
      "AllReduce",
      "And",
      "Any",
      "ArgMax",
      "ArgMin",
      "Asin",
      "Atan",
      "AvgPoolBackprop",
      "BatchMatMul",
      "BatchNormTraining",
      "BatchNormTrainingBackprop",
      "BroadcastDistributed",
      "Ceiling",
      "Convert",
      "ConvolutionBackpropData",
      "ConvolutionBackpropFilters",
      "Cos",
      "Cosh",
      "Dequantize",
      "DynBroadcast",
      "DynPad",
      "DynReshape",
      "DynSlice",
      "EmbeddingLookup",
      "Equal",
      "Erf",
      "Floor",
      "Gather",
      "GatherND",
      "GenerateMask",
      "GetOutputElement",
      "Greater",
      "GreaterEq",
      "Less",
      "LessEq",
      "Log",
      "LRN",
      "Maximum",
      "MaxPoolBackprop",
      "Min",
      "Not",
      "NotEqual",
      "OneHot",
      "Or",
      "Product",
      "Quantize",
      "QuantizedAvgPool",
      "QuantizedConvolutionBias",
      "QuantizedConvolutionBiasAdd",
      "QuantizedConvolutionBiasSignedAdd",
      "QuantizedConvolutionRelu",
      "QuantizedConvolution",
      "QuantizedDot",
      "QuantizedDotBias",
      "QuantizedMaxPool",
      "Send",
      "Recv",
      "Range",
      "ReluBackprop",
      "ReplaceSlice",
      "ReverseSequence",
      "ScatterAdd",
      "ScatterNDAdd",
      "Select",
      "ShapeOf",
      "Sigmoid",
      "SigmoidBackprop",
      "Sign",
      "Sin",
      "Sinh",
      "Sqrt",
      "StopGradient",
      "Tan",
      "Tanh",
      "Tile",
      "TopK",
      "Transpose"};
};

}  // namespace he
}  // namespace ngraph
