/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <assert.h>
#include <memory>

#include "gtest/gtest.h"
#include "heaan/heaan.hpp"
#include "test_util.hpp"

using namespace std;
using namespace ngraph;
using namespace heaan;

TEST(heaan_example, trivial)
{
    int a = 1;
    int b = 2;
    EXPECT_EQ(3, a + b);
}

TEST(heaan_example, basic)
{
    long logN = 13;
    long logQ = 65;
    long logp = 30;
    long logSlots = 3;

    TimeUtils timeutils;
    Context context(logN, logQ);
    SecretKey secretKey(logN);
    Scheme scheme(secretKey, context);

    SetNumThreads(1);
    srand(2018);

    long slots = (1 << logSlots);

    vector<complex<double>> mvec1 = EvaluatorUtils::randomComplexArray(slots);
    vector<complex<double>> mvec2 = EvaluatorUtils::randomComplexArray(slots);
    vector<complex<double>> cvec = EvaluatorUtils::randomComplexArray(slots);

    vector<complex<double>> mvecAdd(slots);
    vector<complex<double>> mvecMult(slots);
    vector<complex<double>> mvecCMult(slots);

    for (long i = 0; i < slots; i++)
    {
        mvecAdd[i] = mvec1[i] + mvec2[i];
        mvecMult[i] = mvec1[i] * mvec2[i];
        mvecCMult[i] = mvec1[i] * cvec[i];
    }

    Ciphertext cipher1 = scheme.encrypt(mvec1, logp, logQ);
    Ciphertext cipher2 = scheme.encrypt(mvec2, logp, logQ);
    Ciphertext addCipher = scheme.add(cipher1, cipher2);

    Ciphertext multCipher = scheme.mult(cipher1, cipher2);
    scheme.reScaleByAndEqual(multCipher, logp);

    Ciphertext cmultCipher = scheme.multByConstVec(cipher1, cvec, slots, logp);
    scheme.reScaleByAndEqual(cmultCipher, logp);

    vector<complex<double>> dvecAdd = scheme.decrypt(secretKey, addCipher);
    vector<complex<double>> dvecMult = scheme.decrypt(secretKey, multCipher);
    vector<complex<double>> dvecCMult = scheme.decrypt(secretKey, cmultCipher);

    EXPECT_EQ(1, 1);

    EXPECT_TRUE((all_close(mvecAdd, dvecAdd)));
    EXPECT_TRUE((all_close(mvecAdd, dvecAdd)));
}
