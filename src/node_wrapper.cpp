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

#include "node_wrapper.hpp"

#include <string>
#include <unordered_map>
#include <utility>

#include "ngraph/op/abs.hpp"
#include "ngraph/op/acos.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/all.hpp"
#include "ngraph/op/allreduce.hpp"
#include "ngraph/op/and.hpp"
#include "ngraph/op/any.hpp"
#include "ngraph/op/argmax.hpp"
#include "ngraph/op/argmin.hpp"
#include "ngraph/op/asin.hpp"
#include "ngraph/op/atan.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/broadcast_distributed.hpp"
#include "ngraph/op/ceiling.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/cos.hpp"
#include "ngraph/op/cosh.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/embedding_lookup.hpp"
#include "ngraph/op/equal.hpp"
#include "ngraph/op/erf.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/experimental/batch_mat_mul.hpp"
#include "ngraph/op/experimental/dyn_broadcast.hpp"
#include "ngraph/op/experimental/dyn_pad.hpp"
#include "ngraph/op/experimental/dyn_replace_slice.hpp"
#include "ngraph/op/experimental/dyn_reshape.hpp"
#include "ngraph/op/experimental/dyn_slice.hpp"
#include "ngraph/op/experimental/generate_mask.hpp"
#include "ngraph/op/experimental/quantized_avg_pool.hpp"
#include "ngraph/op/experimental/quantized_concat.hpp"
#include "ngraph/op/experimental/quantized_conv_bias.hpp"
#include "ngraph/op/experimental/quantized_conv_relu.hpp"
#include "ngraph/op/experimental/quantized_dot.hpp"
#include "ngraph/op/experimental/quantized_dot_bias.hpp"
#include "ngraph/op/experimental/quantized_max_pool.hpp"
#include "ngraph/op/experimental/range.hpp"
#include "ngraph/op/experimental/shape_of.hpp"
#include "ngraph/op/experimental/tile.hpp"
#include "ngraph/op/experimental/transpose.hpp"
#include "ngraph/op/floor.hpp"
#include "ngraph/op/gather.hpp"
#include "ngraph/op/gather_nd.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/greater_eq.hpp"
#include "ngraph/op/less.hpp"
#include "ngraph/op/less_eq.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/lrn.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/not.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/or.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/passthrough.hpp"
#include "ngraph/op/power.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/quantized_convolution.hpp"
#include "ngraph/op/recv.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/reverse_sequence.hpp"
#include "ngraph/op/scatter_add.hpp"
#include "ngraph/op/scatter_nd_add.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/send.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/sign.hpp"
#include "ngraph/op/sin.hpp"
#include "ngraph/op/sinh.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/stop_gradient.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/tan.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/op/topk.hpp"
#include "ngraph/op/xor.hpp"
#include "op/bounded_relu.hpp"

namespace ngraph::he {

NodeWrapper::NodeWrapper(std::shared_ptr<const ngraph::Node> node)
    : m_node{std::move(node)} {
// This expands the op list in op_tbl.hpp into a list of enumerations that look
// like this:
// {"Abs", ngraph::he::OP_TYPEID::Abs},
// {"Acos", ngraph::he::OP_TYPEID::Acos},
// ...
#define NGRAPH_OP(a, b) {#a, ngraph::he::OP_TYPEID::a},
  static std::unordered_map<std::string, ngraph::he::OP_TYPEID> typeid_map{
#include "ngraph/op/op_tbl.hpp"
      NGRAPH_OP(BoundedRelu, ngraph::op)};
#undef NGRAPH_OP
  auto it = typeid_map.find(m_node->description());
  if (it != typeid_map.end()) {
    m_typeid = it->second;
  } else {
    throw unsupported_op("Unsupported op '" + m_node->description() + "'");
  }
}

std::shared_ptr<const op::Op> NodeWrapper::get_op() const {
  if (!get_node()->is_op()) {
    throw ngraph_error("node is not an op");
  }
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
  switch (m_typeid) {
    case OP_TYPEID::Abs: {
      return std::static_pointer_cast<const op::Abs>(m_node);
    }
    case OP_TYPEID::Acos: {
      return std::static_pointer_cast<const op::Acos>(m_node);
    }
    case OP_TYPEID::Add: {
      return std::static_pointer_cast<const op::Add>(m_node);
    }
    case OP_TYPEID::All: {
      return std::static_pointer_cast<const op::All>(m_node);
    }
    case OP_TYPEID::AllReduce: {
      return std::static_pointer_cast<const op::AllReduce>(m_node);
    }
    case OP_TYPEID::And: {
      return std::static_pointer_cast<const op::And>(m_node);
    }
    case OP_TYPEID::Any: {
      return std::static_pointer_cast<const op::Any>(m_node);
    }
    case OP_TYPEID::ArgMax: {
      return std::static_pointer_cast<const op::ArgMax>(m_node);
    }
    case OP_TYPEID::ArgMin: {
      return std::static_pointer_cast<const op::ArgMin>(m_node);
    }
    case OP_TYPEID::Asin: {
      return std::static_pointer_cast<const op::Asin>(m_node);
    }
    case OP_TYPEID::Atan: {
      return std::static_pointer_cast<const op::Atan>(m_node);
    }
    case OP_TYPEID::AvgPool: {
      return std::static_pointer_cast<const op::AvgPool>(m_node);
    }
    case OP_TYPEID::AvgPoolBackprop: {
      return std::static_pointer_cast<const op::AvgPoolBackprop>(m_node);
    }
    case OP_TYPEID::BatchMatMul: {
      return std::static_pointer_cast<const op::BatchMatMul>(m_node);
    }
    case OP_TYPEID::BatchNormInference: {
      return std::static_pointer_cast<const op::BatchNormInference>(m_node);
    }
    case OP_TYPEID::BatchNormTraining: {
      return std::static_pointer_cast<const op::BatchNormTraining>(m_node);
    }
    case OP_TYPEID::BatchNormTrainingBackprop: {
      return std::static_pointer_cast<const op::BatchNormTrainingBackprop>(
          m_node);
    }
    case OP_TYPEID::Broadcast: {
      return std::static_pointer_cast<const op::Broadcast>(m_node);
    }
    case OP_TYPEID::BroadcastDistributed: {
      return std::static_pointer_cast<const op::BroadcastDistributed>(m_node);
    }
    case OP_TYPEID::BroadcastLike: {
      return std::static_pointer_cast<const op::BroadcastLike>(m_node);
    }
    case OP_TYPEID::BoundedRelu: {
      return std::static_pointer_cast<const op::BoundedRelu>(m_node);
    }
    case OP_TYPEID::Ceiling: {
      return std::static_pointer_cast<const op::Ceiling>(m_node);
    }
    case OP_TYPEID::Concat: {
      return std::static_pointer_cast<const op::Concat>(m_node);
    }
    case OP_TYPEID::Constant: {
      throw ngraph_error("Constant is not op");
    }
    case OP_TYPEID::Convert: {
      return std::static_pointer_cast<const op::Convert>(m_node);
    }
    case OP_TYPEID::Convolution: {
      return std::static_pointer_cast<const op::Convolution>(m_node);
    }
    case OP_TYPEID::ConvolutionBackpropData: {
      return std::static_pointer_cast<const op::ConvolutionBackpropData>(
          m_node);
    }
    case OP_TYPEID::ConvolutionBackpropFilters: {
      return std::static_pointer_cast<const op::ConvolutionBackpropFilters>(
          m_node);
    }
    case OP_TYPEID::Cos: {
      return std::static_pointer_cast<const op::Cos>(m_node);
    }
    case OP_TYPEID::Cosh: {
      return std::static_pointer_cast<const op::Cosh>(m_node);
    }
    case OP_TYPEID::Dequantize: {
      return std::static_pointer_cast<const op::Dequantize>(m_node);
    }
    case OP_TYPEID::Divide: {
      return std::static_pointer_cast<const op::Divide>(m_node);
    }
    case OP_TYPEID::Dot: {
      return std::static_pointer_cast<const op::Dot>(m_node);
    }
    case OP_TYPEID::DynBroadcast: {
      return std::static_pointer_cast<const op::DynBroadcast>(m_node);
    }
    case OP_TYPEID::DynPad: {
      return std::static_pointer_cast<const op::DynPad>(m_node);
    }
    case OP_TYPEID::DynReplaceSlice: {
      return std::static_pointer_cast<const op::DynReplaceSlice>(m_node);
    }
    case OP_TYPEID::DynReshape: {
      return std::static_pointer_cast<const op::DynReshape>(m_node);
    }
    case OP_TYPEID::DynSlice: {
      return std::static_pointer_cast<const op::DynSlice>(m_node);
    }
    case OP_TYPEID::EmbeddingLookup: {
      return std::static_pointer_cast<const op::EmbeddingLookup>(m_node);
    }
    case OP_TYPEID::Equal: {
      return std::static_pointer_cast<const op::Equal>(m_node);
    }
    case OP_TYPEID::Erf: {
      return std::static_pointer_cast<const op::Erf>(m_node);
    }
    case OP_TYPEID::Exp: {
      return std::static_pointer_cast<const op::Exp>(m_node);
    }
    case OP_TYPEID::Floor: {
      return std::static_pointer_cast<const op::Floor>(m_node);
    }
    case OP_TYPEID::Gather: {
      return std::static_pointer_cast<const op::Gather>(m_node);
    }
    case OP_TYPEID::GatherND: {
      return std::static_pointer_cast<const op::GatherND>(m_node);
    }
    case OP_TYPEID::GenerateMask: {
      return std::static_pointer_cast<const op::GenerateMask>(m_node);
    }
    case OP_TYPEID::GetOutputElement: {
      return std::static_pointer_cast<const op::GetOutputElement>(m_node);
    }
    case OP_TYPEID::Greater: {
      return std::static_pointer_cast<const op::Greater>(m_node);
    }
    case OP_TYPEID::GreaterEq: {
      return std::static_pointer_cast<const op::GreaterEq>(m_node);
    }
    case OP_TYPEID::Less: {
      return std::static_pointer_cast<const op::Less>(m_node);
    }
    case OP_TYPEID::LessEq: {
      return std::static_pointer_cast<const op::LessEq>(m_node);
    }
    case OP_TYPEID::Log: {
      return std::static_pointer_cast<const op::Log>(m_node);
    }
    case OP_TYPEID::LRN: {
      return std::static_pointer_cast<const op::LRN>(m_node);
    }
    case OP_TYPEID::Max: {
      return std::static_pointer_cast<const op::Max>(m_node);
    }
    case OP_TYPEID::Maximum: {
      return std::static_pointer_cast<const op::Maximum>(m_node);
    }
    case OP_TYPEID::MaxPool: {
      return std::static_pointer_cast<const op::MaxPool>(m_node);
    }
    case OP_TYPEID::MaxPoolBackprop: {
      return std::static_pointer_cast<const op::MaxPoolBackprop>(m_node);
    }
    case OP_TYPEID::Min: {
      return std::static_pointer_cast<const op::Min>(m_node);
    }
    case OP_TYPEID::Minimum: {
      return std::static_pointer_cast<const op::Minimum>(m_node);
    }
    case OP_TYPEID::Multiply: {
      return std::static_pointer_cast<const op::Multiply>(m_node);
    }
    case OP_TYPEID::Negative: {
      return std::static_pointer_cast<const op::Negative>(m_node);
    }
    case OP_TYPEID::Not: {
      return std::static_pointer_cast<const op::Not>(m_node);
    }
    case OP_TYPEID::NotEqual: {
      return std::static_pointer_cast<const op::NotEqual>(m_node);
    }
    case OP_TYPEID::OneHot: {
      return std::static_pointer_cast<const op::OneHot>(m_node);
    }
    case OP_TYPEID::Or: {
      return std::static_pointer_cast<const op::Or>(m_node);
    }
    case OP_TYPEID::Pad: {
      return std::static_pointer_cast<const op::Pad>(m_node);
    }
    case OP_TYPEID::Parameter: {
      return std::static_pointer_cast<const op::Parameter>(m_node);
    }
    case OP_TYPEID::Passthrough: {
      return std::static_pointer_cast<const op::Passthrough>(m_node);
    }
    case OP_TYPEID::Power: {
      return std::static_pointer_cast<const op::Power>(m_node);
    }
    case OP_TYPEID::Product: {
      return std::static_pointer_cast<const op::Product>(m_node);
    }
    case OP_TYPEID::Quantize: {
      return std::static_pointer_cast<const op::Quantize>(m_node);
    }
    case OP_TYPEID::QuantizedAvgPool: {
      throw ngraph_error("Quantized AvgPool unsupported");
    }
    case OP_TYPEID::QuantizedConvolution: {
      return std::static_pointer_cast<const op::QuantizedConvolution>(m_node);
    }
    case OP_TYPEID::QuantizedConvolutionBias: {
      return std::static_pointer_cast<const op::QuantizedConvolutionBias>(
          m_node);
    }
    case OP_TYPEID::QuantizedConvolutionBiasAdd: {
      return std::static_pointer_cast<const op::QuantizedConvolutionBiasAdd>(
          m_node);
    }
    case OP_TYPEID::QuantizedConvolutionBiasSignedAdd: {
      return std::static_pointer_cast<
          const op::QuantizedConvolutionBiasSignedAdd>(m_node);
    }
    case OP_TYPEID::QuantizedConvolutionRelu: {
      return std::static_pointer_cast<const op::QuantizedConvolutionRelu>(
          m_node);
    }
    case OP_TYPEID::QuantizedDot: {
      return std::static_pointer_cast<const op::QuantizedDot>(m_node);
    }
    case OP_TYPEID::QuantizedDotBias: {
      return std::static_pointer_cast<const op::QuantizedDotBias>(m_node);
    }
    case OP_TYPEID::QuantizedMaxPool: {
      return std::static_pointer_cast<const op::QuantizedMaxPool>(m_node);
    }
    case OP_TYPEID::Recv: {
      return std::static_pointer_cast<const op::Recv>(m_node);
    }
    case OP_TYPEID::Range: {
      return std::static_pointer_cast<const op::Range>(m_node);
    }
    case OP_TYPEID::Relu: {
      return std::static_pointer_cast<const op::Relu>(m_node);
    }
    case OP_TYPEID::ReluBackprop: {
      return std::static_pointer_cast<const op::ReluBackprop>(m_node);
    }
    case OP_TYPEID::ReplaceSlice: {
      return std::static_pointer_cast<const op::ReplaceSlice>(m_node);
    }
    case OP_TYPEID::Reshape: {
      return std::static_pointer_cast<const op::Reshape>(m_node);
    }
    case OP_TYPEID::Result: {
      return std::static_pointer_cast<const op::Result>(m_node);
    }
    case OP_TYPEID::Reverse: {
      return std::static_pointer_cast<const op::Reverse>(m_node);
    }
    case OP_TYPEID::ReverseSequence: {
      return std::static_pointer_cast<const op::ReverseSequence>(m_node);
    }
    case OP_TYPEID::ScalarConstantLike: {
      throw ngraph_error("ScalarConstantLike is not op");
    }
    case OP_TYPEID::ScatterAdd: {
      return std::static_pointer_cast<const op::ScatterAdd>(m_node);
    }
    case OP_TYPEID::ScatterNDAdd: {
      return std::static_pointer_cast<const op::ScatterNDAdd>(m_node);
    }
    case OP_TYPEID::Select: {
      return std::static_pointer_cast<const op::Select>(m_node);
    }
    case OP_TYPEID::Send: {
      return std::static_pointer_cast<const op::Send>(m_node);
    }
    case OP_TYPEID::ShapeOf: {
      return std::static_pointer_cast<const op::ShapeOf>(m_node);
    }
    case OP_TYPEID::Sigmoid: {
      return std::static_pointer_cast<const op::Sigmoid>(m_node);
    }
    case OP_TYPEID::SigmoidBackprop: {
      return std::static_pointer_cast<const op::SigmoidBackprop>(m_node);
    }
    case OP_TYPEID::Sign: {
      return std::static_pointer_cast<const op::Sign>(m_node);
    }
    case OP_TYPEID::Sin: {
      return std::static_pointer_cast<const op::Sin>(m_node);
    }
    case OP_TYPEID::Sinh: {
      return std::static_pointer_cast<const op::Sinh>(m_node);
    }
    case OP_TYPEID::Slice: {
      return std::static_pointer_cast<const op::Slice>(m_node);
    }
    case OP_TYPEID::Softmax: {
      return std::static_pointer_cast<const op::Softmax>(m_node);
    }
    case OP_TYPEID::Sqrt: {
      return std::static_pointer_cast<const op::Sqrt>(m_node);
    }
    case OP_TYPEID::StopGradient: {
      return std::static_pointer_cast<const op::StopGradient>(m_node);
    }
    case OP_TYPEID::Subtract: {
      return std::static_pointer_cast<const op::Subtract>(m_node);
    }
    case OP_TYPEID::Sum: {
      return std::static_pointer_cast<const op::Sum>(m_node);
    }
    case OP_TYPEID::Tan: {
      return std::static_pointer_cast<const op::Tan>(m_node);
    }
    case OP_TYPEID::Tanh: {
      return std::static_pointer_cast<const op::Tanh>(m_node);
    }
    case OP_TYPEID::Tile: {
      return std::static_pointer_cast<const op::Tile>(m_node);
    }
    case OP_TYPEID::TopK: {
      return std::static_pointer_cast<const op::TopK>(m_node);
    }
    case OP_TYPEID::Transpose: {
      return std::static_pointer_cast<const op::Transpose>(m_node);
    }
    case OP_TYPEID::Xor: {
      return std::static_pointer_cast<const op::Xor>(m_node);
    }
  }
#pragma GCC diagnostic push
}

}  // namespace ngraph::he
