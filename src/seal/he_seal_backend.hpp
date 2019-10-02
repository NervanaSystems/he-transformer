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
#include "util.hpp"

namespace ngraph {
namespace runtime {
class BackendConstructor;
}
namespace he {
class HESealCipherTensor;
class SealCiphertextWrapper;
/// \brief Class representing a backend using the CKKS homomorphic encryption
/// scheme.
class HESealBackend : public ngraph::runtime::Backend {
 public:
  /// \brief Constructs a backend with default parameter choice
  HESealBackend();
  /// \brief Constructs a backend with the given encryption parameters
  /// \param[in] parms Encryption parameters
  HESealBackend(const ngraph::he::HESealEncryptionParameters& parms);

  /// \brief Prepares the backend with the encryption context, including
  /// generating encryption keys, encryptor, decryptor, evaluator, and encoder
  void generate_context();

  /// \brief Constructs a plaintext tensor
  /// \param[in] element_type Datatype to store in the tensor
  /// \param[in] shape Shape of the tensor
  std::shared_ptr<runtime::Tensor> create_tensor(
      const element::Type& element_type, const Shape& shape) override;

  /// \brief Unimplemented
  inline std::shared_ptr<runtime::Tensor> create_tensor(
      const element::Type& element_type, const Shape& shape,
      void* memory_pointer) override {
    throw ngraph_error("create_tensor unimplemented");
  }

  /// \brief Compiles a function
  /// \brief param[in] function Function to compile
  /// \brief param[in] enable_performance_data TODO: unused
  /// \returns An executable object
  std::shared_ptr<ngraph::runtime::Executable> compile(
      std::shared_ptr<Function> function,
      bool enable_performance_data = false) override;

  /// \brief Returns whether or not a given operation is supported
  /// \param[in] node Node representing an operation
  bool is_supported(const Node& node) const override;

  /// \brief Sets a configuration for the backend
  /// \paran[in] config Configuration map. It should contain entries in one of
  /// three forms:
  ///     1) {tensor_name : "client_input"}, which indicates the specified
  ///     tensor should be loaded from the client. Note, the tensor may or may
  ///     not be encrypted, as determined by the client.
  ///      2) {enable_client : "True" /"False"}, which indicates whether or not
  ///      the client should be
  ///     enabled
  ///     3) {tensor_name : "encrypt"}, which indicates the specified
  ///     tensor should be encrypted. By default, tensors may or may not be
  ///     encrypted. Setting this option will encrypt the plaintext tensor of
  ///     name tensor_name if not already encrypted.
  ///
  ///  \warning Specfying entries of form 1) without an entry of form 2) will
  ///  not load the tensors from the client
  /// \param[out] error Error string. Unused
  bool set_config(const std::map<std::string, std::string>& config,
                  std::string& error) override;

  /// \brief Returns whether or not a given datatype is supported
  /// \param[in] type Datatype
  /// \returns True if datatype is supported, false otherwise
  inline bool is_supported_type(const ngraph::element::Type& type) const {
    return m_supported_element_types.find(type.hash()) !=
           m_supported_element_types.end();
  }

  /// \brief Creates a cipher tensor using plaintext packing along the batch
  /// (i.e. first) axis
  /// \param[in] element_type Datatype stored in the tensor
  /// \param[in] shape Shape of the tensor
  /// \returns Pointer to created tensor
  std::shared_ptr<runtime::Tensor> create_packed_cipher_tensor(
      const element::Type& element_type, const Shape& shape);

  /// \brief Creates a plaintext tensor using plaintext packing along the batch
  /// (i.e. first) axis
  /// \param[in] element_type Datatype stored in the tensor
  /// \param[in] shape Shape of the tensor
  /// \returns Pointer to created tensor
  std::shared_ptr<runtime::Tensor> create_packed_plain_tensor(
      const element::Type& element_type, const Shape& shape);

  /// \brief Creates a plaintext tensor
  /// \param[in] element_type Datatype stored in the tensor
  /// \param[in] shape Shape of the tensor
  /// \param[in] packed Whether or not to use plaintext packing
  /// \returns Pointer to created tensor
  std::shared_ptr<runtime::Tensor> create_plain_tensor(
      const element::Type& element_type, const Shape& shape,
      const bool packed = false) const;

  /// \brief Creates a ciphertext tensor
  /// \param[in] element_type Datatype stored in the tensor
  /// \param[in] shape Shape of the tensor
  /// \param[in] packed Whether or not to use plaintext packing
  /// \param[in] name Name of the created tensor
  /// \returns Pointer to created tensor
  std::shared_ptr<runtime::Tensor> create_cipher_tensor(
      const element::Type& element_type, const Shape& shape,
      const bool packed = false, const std::string& name = "external") const;

  /// \brief Creates ciphertext with given value
  /// \param[in] value Value to encode and encrypt
  /// \param[in] element_type Datatype of the values to store
  /// \param[in] batch_size TODO: remove
  /// \returns Pointer to created ciphertext
  std::shared_ptr<ngraph::he::SealCiphertextWrapper> create_valued_ciphertext(
      float value, const element::Type& element_type,
      size_t batch_size = 1) const;

  /// \brief Creates empty ciphertext with backend's complex packing status
  /// \returns Pointer to created ciphertext
  inline std::shared_ptr<ngraph::he::SealCiphertextWrapper>
  create_empty_ciphertext() const {
    return std::make_shared<ngraph::he::SealCiphertextWrapper>(
        complex_packing());
  }

  /// \brief Creates empty ciphertext
  /// \param[in] complex_packing Whether or not ciphertext uses complex packing
  /// \returns Pointer to created ciphertext
  static inline std::shared_ptr<ngraph::he::SealCiphertextWrapper>
  create_empty_ciphertext(bool complex_packing) {
    return std::make_shared<ngraph::he::SealCiphertextWrapper>(complex_packing);
  }

  /// \brief Creates empty ciphertext at given parameter choice with backend's
  /// complex packing status
  /// \param[in] parms_id Seal encryption parameter id
  /// \returns Pointer to created ciphertext
  inline std::shared_ptr<ngraph::he::SealCiphertextWrapper>
  create_empty_ciphertext(seal::parms_id_type parms_id) const {
    return std::make_shared<ngraph::he::SealCiphertextWrapper>(
        seal::Ciphertext(m_context, parms_id), complex_packing());
  }

  /// \brief Creates empty ciphertext at given parameter choice with backend's
  /// complex packing status
  /// \param[in] pool Memory pool used for new memory allocation
  /// \returns Pointer to created ciphertext
  inline std::shared_ptr<ngraph::he::SealCiphertextWrapper>
  create_empty_ciphertext(const seal::MemoryPoolHandle& pool) const {
    return std::make_shared<ngraph::he::SealCiphertextWrapper>(
        pool, complex_packing());
  }

  /// \brief Creates SEAL context from encryption parameters
  /// \param[in] sp Pointer to encrpytion parameters
  /// \returns pointer to created SEAL context
  std::shared_ptr<seal::SEALContext> make_seal_context(
      const std::shared_ptr<ngraph::he::HESealEncryptionParameters> sp);

  /// \brief TODO
  void decode(void* output, const ngraph::he::HEPlaintext& input,
              const element::Type& type, size_t count = 1) const;

  /// \brief TODO
  void decode(ngraph::he::HEPlaintext& output,
              const ngraph::he::SealPlaintextWrapper& input) const;

  /// \brief TODO
  void encrypt(std::shared_ptr<ngraph::he::SealCiphertextWrapper>& output,
               const ngraph::he::HEPlaintext& input,
               const element::Type& element_type,
               bool complex_packing = false) const;

  /// \brief TODO
  void decrypt(ngraph::he::HEPlaintext& output,
               const SealCiphertextWrapper& input) const;

  /// \brief Returns pointer to SEAL context
  const inline std::shared_ptr<seal::SEALContext> get_context() const {
    return m_context;
  }

  /// \brief Returns pointer to secret key
  const inline std::shared_ptr<seal::SecretKey> get_secret_key() const {
    return m_secret_key;
  }

  /// \brief Returns pointer to public key
  const inline std::shared_ptr<seal::PublicKey> get_public_key() const {
    return m_public_key;
  }

  /// \brief Returns pointer to relinearization keys
  const inline std::shared_ptr<seal::RelinKeys> get_relin_keys() const {
    return m_relin_keys;
  }

  /// \brief Returns pointer to Galois keys
  const inline std::shared_ptr<seal::GaloisKeys> get_galois_keys() const {
    return m_galois_keys;
  }

  /// \brief Returns pointer to encryptor
  const inline std::shared_ptr<seal::Encryptor> get_encryptor() const {
    return m_encryptor;
  }

  /// \brief Returns pointer to decryptor
  const inline std::shared_ptr<seal::Decryptor> get_decryptor() const {
    return m_decryptor;
  }

  /// \brief Retursn a pointer to evaluator
  const inline std::shared_ptr<seal::Evaluator> get_evaluator() const {
    return m_evaluator;
  }

  /// \brief Returns the encryption parameters
  const ngraph::he::HESealEncryptionParameters& get_encryption_parameters()
      const {
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

  /// \brief TODO
  const std::unordered_map<std::uint64_t, std::uint64_t>& barrett64_ratio_map()
      const {
    return m_barrett64_ratio_map;
  }

  /// \brief Returns the top-level scale used for encoding
  inline double get_scale() const { return m_encryption_params.scale(); }

  /// \brief Sets plaintext packing
  /// TODO: rename to plaintext_pack_data
  void set_pack_data(const bool pack) { m_pack_data = pack; }

  /// \brief Returns whether or not complex packing is used
  bool complex_packing() const { return m_encryption_params.complex_packing(); }

  /// \brief Returns whether or not plaintext packing is used
  bool pack_data() const { return m_pack_data; }

  /// \brief Returns whether or not the model is encrypted
  bool encrypt_model() const { return m_encrypt_model; }

  /// \brief Returns whether or not the rescaling operation is performed after
  /// every multiplication.
  /// \warning Naive rescaling results in a dramatic performance penalty for
  /// Convolution and Dot operations. Typically, this should never be used
  bool naive_rescaling() const { return m_naive_rescaling; }

  /// \brief Returns whether or not the rescaling operation is performed after
  /// every multiplication.
  /// \warning Naive rescaling results in a dramatic performance penalty for
  /// Convolution and Dot operations. Typically, this should never be used
  bool& naive_rescaling() { return m_naive_rescaling; }

  /// \brief Returns the chain index, also known as level, of the ciphertext
  /// \param[in] cipher Ciphertext whose chain index to return
  /// \returns The chain index of the ciphertext.
  inline size_t get_chain_index(const SealCiphertextWrapper& cipher) const {
    return m_context->get_context_data(cipher.ciphertext().parms_id())
        ->chain_index();
  }

  /// \brief Returns the chain index, also known as level, of the plaintext
  /// \param[in] plain Plaintext whose chain index to return
  /// \returns The chain index of the ciphertext.
  inline size_t get_chain_index(const SealPlaintextWrapper& plain) const {
    return m_context->get_context_data(plain.plaintext().parms_id())
        ->chain_index();
  }

  /// \brief Returns set of tensors to be provided by the client
  inline std::unordered_set<std::string> get_client_tensor_names() const {
    return m_client_tensor_names;
  }

  /// \brief Returns set of tensors to be encrypted
  inline std::unordered_set<std::string> get_encrypted_tensor_names() const {
    return m_encrypted_tensor_names;
  }

 private:
  bool m_pack_data{
      !ngraph::he::flag_to_bool(std::getenv("NGRAPH_UNPACK_DATA"))};
  bool m_encrypt_model{
      ngraph::he::flag_to_bool(std::getenv("NGRAPH_ENCRYPT_MODEL"))};
  bool m_naive_rescaling{
      ngraph::he::flag_to_bool(std::getenv("NAIVE_RESCALING"))};
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

  std::unordered_set<size_t> m_supported_element_types{
      element::f32.hash(), element::i64.hash(), element::f64.hash()};

  std::unordered_set<std::string> m_client_tensor_names;
  std::unordered_set<std::string> m_encrypted_tensor_names;

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
      "Divide",
      "DynBroadcast",
      "DynPad",
      "DynReshape",
      "DynSlice",
      "EmbeddingLookup",
      "Equal",
      "Erf",
      "Exp",
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
      "Max",
      "Maximum",
      "MaxPoolBackprop",
      "Min",
      "Not",
      "NotEqual",
      "OneHot",
      "Or",
      "Power",
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
      "Softmax",
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
