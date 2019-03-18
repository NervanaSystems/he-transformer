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

#include <ostream>

namespace ngraph {
namespace runtime {
namespace he {
class HEEncryptionParameters {
 public:
  HEEncryptionParameters(){};
  virtual ~HEEncryptionParameters(){};

  virtual void save(std::ostream& stream) const = 0;

  virtual void set_poly_modulus_degree(size_t poly_modulus_degree) = 0;

  virtual void set_coeff_modulus(
      const std::vector<std::uint64_t>& coeff_modulus) = 0;
};
}  // namespace he
}  // namespace runtime
}  // namespace ngraph
