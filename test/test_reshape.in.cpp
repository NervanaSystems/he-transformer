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

#include "he_backend.hpp"
#include "ngraph/ngraph.hpp"
#include "test_util.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, reshape_t2v_012) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  Shape shape_a{2, 2, 3};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{12};
  auto r = make_shared<op::Reshape>(A, AxisVector{0, 1, 2}, shape_r);
  auto f = make_shared<Function>(r, ParameterVector{A});
  // Create some tensors for input/output
  auto tensors_list =
      generate_plain_cipher_tensors({r}, {A}, backend.get(), true);
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    backend->call(backend->compile(f), {result}, {a});
    EXPECT_TRUE(
        all_close((vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
                  read_vector<float>(result), 1e-3f));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_t2s_012) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  Shape shape_a{1, 1, 1};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{};
  auto r = make_shared<op::Reshape>(A, AxisVector{0, 1, 2}, shape_r);
  auto f = make_shared<Function>(r, ParameterVector{A});
  // Create some tensors for input/output
  auto tensors_list =
      generate_plain_cipher_tensors({r}, {A}, backend.get(), true);
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a, vector<float>{6});

    backend->call(backend->compile(f), {result}, {a});
    EXPECT_TRUE(
        all_close((vector<float>{6}), read_vector<float>(result), 1e-3f));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_t2s_120) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  Shape shape_a{1, 1, 1};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{};
  auto r = make_shared<op::Reshape>(A, AxisVector{0, 1, 2}, shape_r);
  auto f = make_shared<Function>(r, ParameterVector{A});
  // Create some tensors for input/output
  auto tensors_list =
      generate_plain_cipher_tensors({r}, {A}, backend.get(), true);
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a, vector<float>{6});

    backend->call(backend->compile(f), {result}, {a});
    EXPECT_TRUE(
        all_close((vector<float>{6}), read_vector<float>(result), 1e-3f));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_s2t) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  Shape shape_a{};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{1, 1, 1, 1, 1, 1};
  auto r = make_shared<op::Reshape>(A, AxisVector{}, shape_r);
  auto f = make_shared<Function>(r, ParameterVector{A});
  // Create some tensors for input/output
  auto tensors_list =
      generate_plain_cipher_tensors({r}, {A}, backend.get(), true);
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a, vector<float>{42});

    backend->call(backend->compile(f), {result}, {a});
    EXPECT_TRUE(
        all_close((vector<float>{42}), read_vector<float>(result), 1e-3f));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_v2m_col) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  Shape shape_a{3};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{3, 1};
  auto r = make_shared<op::Reshape>(A, AxisVector{0}, shape_r);
  auto f = make_shared<Function>(r, ParameterVector{A});
  // Create some tensors for input/output
  auto tensors_list =
      generate_plain_cipher_tensors({r}, {A}, backend.get(), true);
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a, vector<float>{1, 2, 3});
    backend->call(backend->compile(f), {result}, {a});
    EXPECT_TRUE(
        all_close((vector<float>{1, 2, 3}), read_vector<float>(result), 1e-3f));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_v2m_row) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  Shape shape_a{3};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{1, 3};
  auto r = make_shared<op::Reshape>(A, AxisVector{0}, shape_r);
  auto f = make_shared<Function>(r, ParameterVector{A});
  // Create some tensors for input/output
  auto tensors_list =
      generate_plain_cipher_tensors({r}, {A}, backend.get(), true);
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a, vector<float>{1, 2, 3});
    backend->call(backend->compile(f), {result}, {a});
    EXPECT_TRUE(
        all_close((vector<float>{1, 2, 3}), read_vector<float>(result), 1e-3f));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_v2t_middle) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  Shape shape_a{3};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{1, 3, 1};
  auto r = make_shared<op::Reshape>(A, AxisVector{0}, shape_r);
  auto f = make_shared<Function>(r, ParameterVector{A});
  // Create some tensors for input/output
  auto tensors_list =
      generate_plain_cipher_tensors({r}, {A}, backend.get(), true);
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a, vector<float>{1, 2, 3});
    backend->call(backend->compile(f), {result}, {a});
    EXPECT_TRUE(
        all_close((vector<float>{1, 2, 3}), read_vector<float>(result), 1e-3f));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_m2m_same) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  Shape shape_a{3, 3};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{3, 3};
  auto r = make_shared<op::Reshape>(A, AxisVector{0, 1}, shape_r);
  auto f = make_shared<Function>(r, ParameterVector{A});
  // Create some tensors for input/output
  auto tensors_list =
      generate_plain_cipher_tensors({r}, {A}, backend.get(), true);
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    backend->call(backend->compile(f), {result}, {a});
    EXPECT_TRUE(all_close((vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}),
                          read_vector<float>(result), 1e-3f));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_m2m_transpose) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  Shape shape_a{3, 3};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{3, 3};
  auto r = make_shared<op::Reshape>(A, AxisVector{1, 0}, shape_r);
  auto f = make_shared<Function>(r, ParameterVector{A});
  // Create some tensors for input/output
  auto tensors_list =
      generate_plain_cipher_tensors({r}, {A}, backend.get(), true);
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    backend->call(backend->compile(f), {result}, {a});
    EXPECT_TRUE(all_close((vector<float>{1, 4, 7, 2, 5, 8, 3, 6, 9}),
                          read_vector<float>(result), 1e-3f));
  }
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_m2m_dim_change_transpose) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  Shape shape_a{3, 2};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{2, 3};
  auto r = make_shared<op::Reshape>(A, AxisVector{1, 0}, shape_r);
  auto f = make_shared<Function>(r, ParameterVector{A});
  // Create some tensors for input/output
  auto tensors_list =
      generate_plain_cipher_tensors({r}, {A}, backend.get(), true);
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    backend->call(backend->compile(f), {result}, {a});
    EXPECT_TRUE(all_close((vector<float>{1, 3, 5, 2, 4, 6}),
                          read_vector<float>(result), 1e-3f));
  }
}

//
// Numpy:
//
// >>> x = linspace(1,2*2*3*3*2*4,2*2*3*3*2*4)
// >>> x.shape=(2,2,3,3,2,4)
// >>> y = ascontiguousarray(transpose(x,(2,4,0,5,3,1)))
// >>> y.shape=2*2*3*3*2*4
// >>> y
// array([   1.,   73.,    9.,   81.,   17.,   89.,    2.,   74.,   10.,
//          82.,   18.,   90.,    3.,   75.,   11.,   83.,   19.,   91.,
//           4.,   76.,   12.,   84.,   20.,   92.,  145.,  217.,  153.,
//         225.,  161.,  233.,  146.,  218.,  154.,  226.,  162.,  234.,
//         147.,  219.,  155.,  227.,  163.,  235.,  148.,  220.,  156.,
//         228.,  164.,  236.,    5.,   77.,   13.,   85.,   21.,   93.,
//           6.,   78.,   14.,   86.,   22.,   94.,    7.,   79.,   15.,
//          87.,   23.,   95.,    8.,   80.,   16.,   88.,   24.,   96.,
//         149.,  221.,  157.,  229.,  165.,  237.,  150.,  222.,  158.,
//         230.,  166.,  238.,  151.,  223.,  159.,  231.,  167.,  239.,
//         152.,  224.,  160.,  232.,  168.,  240.,   25.,   97.,   33.,
//         105.,   41.,  113.,   26.,   98.,   34.,  106.,   42.,  114.,
//          27.,   99.,   35.,  107.,   43.,  115.,   28.,  100.,   36.,
//         108.,   44.,  116.,  169.,  241.,  177.,  249.,  185.,  257.,
//         170.,  242.,  178.,  250.,  186.,  258.,  171.,  243.,  179.,
//         251.,  187.,  259.,  172.,  244.,  180.,  252.,  188.,  260.,
//          29.,  101.,   37.,  109.,   45.,  117.,   30.,  102.,   38.,
//         110.,   46.,  118.,   31.,  103.,   39.,  111.,   47.,  119.,
//          32.,  104.,   40.,  112.,   48.,  120.,  173.,  245.,  181.,
//         253.,  189.,  261.,  174.,  246.,  182.,  254.,  190.,  262.,
//         175.,  247.,  183.,  255.,  191.,  263.,  176.,  248.,  184.,
//         256.,  192.,  264.,   49.,  121.,   57.,  129.,   65.,  137.,
//          50.,  122.,   58.,  130.,   66.,  138.,   51.,  123.,   59.,
//         131.,   67.,  139.,   52.,  124.,   60.,  132.,   68.,  140.,
//         193.,  265.,  201.,  273.,  209.,  281.,  194.,  266.,  202.,
//         274.,  210.,  282.,  195.,  267.,  203.,  275.,  211.,  283.,
//         196.,  268.,  204.,  276.,  212.,  284.,   53.,  125.,   61.,
//         133.,   69.,  141.,   54.,  126.,   62.,  134.,   70.,  142.,
//          55.,  127.,   63.,  135.,   71.,  143.,   56.,  128.,   64.,
//         136.,   72.,  144.,  197.,  269.,  205.,  277.,  213.,  285.,
//         198.,  270.,  206.,  278.,  214.,  286.,  199.,  271.,  207.,
//         279.,  215.,  287.,  200.,  272.,  208.,  280.,  216.,  288.])
//
NGRAPH_TEST(${BACKEND_NAME}, reshape_6d) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");

  vector<float> a_data(2 * 2 * 3 * 3 * 2 * 4);
  for (int i = 0; i < 2 * 2 * 3 * 3 * 2 * 4; i++) {
    a_data[i] = float(i + 1);
  }

  Shape shape_a{2, 2, 3, 3, 2, 4};
  auto A = make_shared<op::Parameter>(element::f32, shape_a);
  Shape shape_r{3, 2, 2, 4, 3, 2};

  auto r = make_shared<op::Reshape>(A, AxisVector{2, 4, 0, 5, 3, 1}, shape_r);
  auto f = make_shared<Function>(r, ParameterVector{A});
  // Create some tensors for input/output
  auto tensors_list =
      generate_plain_cipher_tensors({r}, {A}, backend.get(), true);
  for (auto tensors : tensors_list) {
    auto results = get<0>(tensors);
    auto inputs = get<1>(tensors);

    auto a = inputs[0];
    auto result = results[0];

    copy_data(a, a_data);
    backend->call(backend->compile(f), {result}, {a});
    EXPECT_TRUE(all_close(
        (vector<float>{
            1.,   73.,  9.,   81.,  17.,  89.,  2.,   74.,  10.,  82.,  18.,
            90.,  3.,   75.,  11.,  83.,  19.,  91.,  4.,   76.,  12.,  84.,
            20.,  92.,  145., 217., 153., 225., 161., 233., 146., 218., 154.,
            226., 162., 234., 147., 219., 155., 227., 163., 235., 148., 220.,
            156., 228., 164., 236., 5.,   77.,  13.,  85.,  21.,  93.,  6.,
            78.,  14.,  86.,  22.,  94.,  7.,   79.,  15.,  87.,  23.,  95.,
            8.,   80.,  16.,  88.,  24.,  96.,  149., 221., 157., 229., 165.,
            237., 150., 222., 158., 230., 166., 238., 151., 223., 159., 231.,
            167., 239., 152., 224., 160., 232., 168., 240., 25.,  97.,  33.,
            105., 41.,  113., 26.,  98.,  34.,  106., 42.,  114., 27.,  99.,
            35.,  107., 43.,  115., 28.,  100., 36.,  108., 44.,  116., 169.,
            241., 177., 249., 185., 257., 170., 242., 178., 250., 186., 258.,
            171., 243., 179., 251., 187., 259., 172., 244., 180., 252., 188.,
            260., 29.,  101., 37.,  109., 45.,  117., 30.,  102., 38.,  110.,
            46.,  118., 31.,  103., 39.,  111., 47.,  119., 32.,  104., 40.,
            112., 48.,  120., 173., 245., 181., 253., 189., 261., 174., 246.,
            182., 254., 190., 262., 175., 247., 183., 255., 191., 263., 176.,
            248., 184., 256., 192., 264., 49.,  121., 57.,  129., 65.,  137.,
            50.,  122., 58.,  130., 66.,  138., 51.,  123., 59.,  131., 67.,
            139., 52.,  124., 60.,  132., 68.,  140., 193., 265., 201., 273.,
            209., 281., 194., 266., 202., 274., 210., 282., 195., 267., 203.,
            275., 211., 283., 196., 268., 204., 276., 212., 284., 53.,  125.,
            61.,  133., 69.,  141., 54.,  126., 62.,  134., 70.,  142., 55.,
            127., 63.,  135., 71.,  143., 56.,  128., 64.,  136., 72.,  144.,
            197., 269., 205., 277., 213., 285., 198., 270., 206., 278., 214.,
            286., 199., 271., 207., 279., 215., 287., 200., 272., 208., 280.,
            216., 288.}),
        read_vector<float>(result), 1e-3f));
  }
}
