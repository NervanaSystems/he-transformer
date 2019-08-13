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
class HESealBackend : public ngraph::runtime::Backend {
 public:
  HESealBackend();
  HESealBackend(const ngraph::he::HESealEncryptionParameters& sp);

  //
  // ngraph backend overrides
  //
  std::shared_ptr<runtime::Tensor> create_tensor(
      const element::Type& element_type, const Shape& shape) override;

  inline std::shared_ptr<runtime::Tensor> create_tensor(
      const element::Type& element_type, const Shape& shape,
      void* memory_pointer) override {
    throw ngraph_error("create_tensor unimplemented");
  }

  std::shared_ptr<ngraph::runtime::Executable> compile(
      std::shared_ptr<Function> func,
      bool enable_performance_data = false) override;

  bool is_supported(const Node& node) const override;

  inline bool is_supported_type(const ngraph::element::Type& type) const {
    return m_supported_element_types.find(type.hash()) !=
           m_supported_element_types.end();
  }

  void validate_he_call(std::shared_ptr<const Function> function,
                        const std::vector<std::shared_ptr<HETensor>>& outputs,
                        const std::vector<std::shared_ptr<HETensor>>& inputs);

  //
  // Tensor creation
  //
  std::shared_ptr<runtime::Tensor> create_packed_cipher_tensor(
      const element::Type& element_type, const Shape& shape);

  std::shared_ptr<runtime::Tensor> create_packed_plain_tensor(
      const element::Type& element_type, const Shape& shape);

  std::shared_ptr<runtime::Tensor> create_plain_tensor(
      const element::Type& element_type, const Shape& shape,
      const bool packed = false) const;

  std::shared_ptr<runtime::Tensor> create_cipher_tensor(
      const element::Type& element_type, const Shape& shape,
      const bool packed = false, const std::string& name = "external") const;

  //
  // Cipher/plaintext creation
  //
  std::shared_ptr<ngraph::he::SealCiphertextWrapper> create_valued_ciphertext(
      float value, const element::Type& element_type,
      size_t batch_size = 1) const;

  inline std::shared_ptr<ngraph::he::SealCiphertextWrapper>
  create_empty_ciphertext() const {
    return std::make_shared<ngraph::he::SealCiphertextWrapper>(
        m_complex_packing);
  }

  inline std::shared_ptr<ngraph::he::SealCiphertextWrapper>
  create_empty_ciphertext(seal::parms_id_type parms_id) const {
    return std::make_shared<ngraph::he::SealCiphertextWrapper>(
        seal::Ciphertext(m_context, parms_id), m_complex_packing);
  }

  inline std::shared_ptr<ngraph::he::SealCiphertextWrapper>
  create_empty_ciphertext(const seal::MemoryPoolHandle& pool) const {
    return std::make_shared<ngraph::he::SealCiphertextWrapper>(
        pool, m_complex_packing);
  }

  std::shared_ptr<seal::SEALContext> make_seal_context(
      const std::shared_ptr<ngraph::he::HESealEncryptionParameters> sp);

  void decode(void* output, const ngraph::he::HEPlaintext& input,
              const element::Type& type, size_t count = 1) const;

  void decode(ngraph::he::HEPlaintext& output,
              const ngraph::he::SealPlaintextWrapper& input) const;

  void encrypt(std::shared_ptr<ngraph::he::SealCiphertextWrapper>& output,
               const ngraph::he::HEPlaintext& input,
               const element::Type& element_type,
               bool complex_packing = false) const;

  void decrypt(ngraph::he::HEPlaintext& output,
               const SealCiphertextWrapper& input) const;

  const inline std::shared_ptr<seal::SEALContext> get_context() const {
    return m_context;
  }

  const inline std::shared_ptr<seal::SecretKey> get_secret_key() const {
    return m_secret_key;
  }

  const inline std::shared_ptr<seal::PublicKey> get_public_key() const {
    return m_public_key;
  }

  const inline std::shared_ptr<seal::RelinKeys> get_relin_keys() const {
    return m_relin_keys;
  }

  const inline std::shared_ptr<seal::Encryptor> get_encryptor() const {
    return m_encryptor;
  }

  const inline std::shared_ptr<seal::Decryptor> get_decryptor() const {
    return m_decryptor;
  }

  void set_relin_keys(const seal::RelinKeys& keys) {
    m_relin_keys = std::make_shared<seal::RelinKeys>(keys);
  }

  void set_public_key(const seal::PublicKey& key) {
    m_public_key = std::make_shared<seal::PublicKey>(key);
    m_encryptor = std::make_shared<seal::Encryptor>(m_context, *m_public_key);
  }

  const inline std::shared_ptr<seal::Evaluator> get_evaluator() const {
    return m_evaluator;
  }

  const ngraph::he::HESealEncryptionParameters& get_encryption_parameters()
      const {
    return m_encryption_params;
  }

  const std::shared_ptr<seal::CKKSEncoder> get_ckks_encoder() const {
    return m_ckks_encoder;
  }

  const std::unordered_map<std::uint64_t, std::uint64_t>& barrett64_ratio_map()
      const {
    return m_barrett64_ratio_map;
  }

  inline double get_scale() const { return m_scale; }

  void set_pack_data(bool pack) { m_pack_data = pack; }

  bool complex_packing() const { return m_complex_packing; }
  bool& complex_packing() { return m_complex_packing; }

  bool encrypt_data() const { return m_encrypt_data; }
  bool pack_data() const { return m_pack_data; }
  bool encrypt_model() const { return m_encrypt_model; }

  // TODO: remove once performance impact is understood
  bool naive_rescaling() const { return m_naive_rescaling; }
  bool& naive_rescaling() { return m_naive_rescaling; }

 private:
  bool m_encrypt_data{
      ngraph::he::flag_to_bool(std::getenv("NGRAPH_ENCRYPT_DATA"))};
  bool m_pack_data{
      !ngraph::he::flag_to_bool(std::getenv("NGRAPH_UNPACK_DATA"))};
  bool m_encrypt_model{
      ngraph::he::flag_to_bool(std::getenv("NGRAPH_ENCRYPT_MODEL"))};
  bool m_complex_packing{
      ngraph::he::flag_to_bool(std::getenv("NGRAPH_COMPLEX_PACK"))};
  bool m_naive_rescaling{
      ngraph::he::flag_to_bool(std::getenv("NAIVE_RESCALING"))};
  bool m_enable_client{
      ngraph::he::flag_to_bool(std::getenv("NGRAPH_ENABLE_CLIENT"))};

  std::shared_ptr<seal::SecretKey> m_secret_key;
  std::shared_ptr<seal::PublicKey> m_public_key;
  std::shared_ptr<seal::RelinKeys> m_relin_keys;
  std::shared_ptr<seal::Encryptor> m_encryptor;
  std::shared_ptr<seal::Decryptor> m_decryptor;
  std::shared_ptr<seal::SEALContext> m_context;
  std::shared_ptr<seal::Evaluator> m_evaluator;
  std::shared_ptr<seal::KeyGenerator> m_keygen;
  HESealEncryptionParameters m_encryption_params;
  std::shared_ptr<seal::CKKSEncoder> m_ckks_encoder;
  double m_scale;

  // Stores Barrett64 ratios for moduli under 30 bits
  std::unordered_map<std::uint64_t, std::uint64_t> m_barrett64_ratio_map;

  std::unordered_set<size_t> m_supported_element_types{
      element::f32.hash(), element::i64.hash(), element::f64.hash()};

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
