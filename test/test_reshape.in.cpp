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

#include "he_op_annotations.hpp"
#include "ngraph/ngraph.hpp"
#include "seal/he_seal_backend.hpp"
#include "test_util.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;
using namespace ngraph::he;

static string s_manifest = "${MANIFEST}";

auto reshape_test = [](const Shape& shape_a, const Shape& shape_r,
                       const AxisVector& axis_vector,
                       const vector<float>& input, const vector<float>& output,
                       const bool arg1_encrypted, const bool complex_packing,
                       const bool packed) {
  auto backend = runtime::Backend::create("${BACKEND_NAME}");
  auto he_backend = static_cast<HESealBackend*>(backend.get());

  if (complex_packing) {
    he_backend->update_encryption_parameters(
        HESealEncryptionParameters::default_complex_packing_parms());
  }

  auto a = make_shared<op::Parameter>(element::f32, shape_a);
  auto t = make_shared<op::Reshape>(a, axis_vector, shape_r);
  auto f = make_shared<Function>(t, ParameterVector{a});

  a->set_op_annotations(
      test::he::annotation_from_flags(false, arg1_encrypted, packed));

  auto t_a =
      test::he::tensor_from_flags(*he_backend, shape_a, arg1_encrypted, packed);
  auto t_result =
      test::he::tensor_from_flags(*he_backend, shape_r, arg1_encrypted, packed);

  copy_data(t_a, input);

  auto handle = backend->compile(f);
  handle->call_with_validate({t_result}, {t_a});
  EXPECT_TRUE(test::he::all_close(read_vector<float>(t_result), output, 1e-3f));
};

NGRAPH_TEST(${BACKEND_NAME}, reshape_t2v_012_plain_plain) {
  for (bool arg1_encrypted : vector<bool>{false, true}) {
    for (bool complex_packing : vector<bool>{false, true}) {
      for (bool packing : vector<bool>{false, true}) {
        reshape_test(Shape{2, 2, 3}, Shape{12}, AxisVector{0, 1, 2},
                     vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                     vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                     arg1_encrypted, complex_packing, packing);
      }
    }
  }
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_t2s_012_plain_plain) {
  reshape_test(Shape{1, 1, 1}, Shape{}, AxisVector{0, 1, 2}, vector<float>{6},
               vector<float>{6}, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_t2s_120_plain_plain) {
  reshape_test(Shape{1, 1, 1}, Shape{}, AxisVector{0, 1, 2}, vector<float>{6},
               vector<float>{6}, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_s2t_plain_plain) {
  reshape_test(Shape{}, Shape{1, 1, 1, 1, 1, 1}, AxisVector{},
               vector<float>{42}, vector<float>{42}, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_v2m_col_plain_plain) {
  reshape_test(Shape{3}, Shape{3, 1}, AxisVector{0}, vector<float>{1, 2, 3},
               vector<float>{1, 2, 3}, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_v2m_row_plain_plain) {
  reshape_test(Shape{3}, Shape{1, 3}, AxisVector{0}, vector<float>{1, 2, 3},
               vector<float>{1, 2, 3}, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_v2t_middle_plain_plain) {
  reshape_test(Shape{3}, Shape{1, 3, 1}, AxisVector{0}, vector<float>{1, 2, 3},
               vector<float>{1, 2, 3}, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_m2m_same_plain_plain) {
  reshape_test(Shape{3, 3}, Shape{3, 3}, AxisVector{0, 1},
               vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9},
               vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_m2m_transpose_plain_plain) {
  reshape_test(Shape{3, 3}, Shape{3, 3}, AxisVector{1, 0},
               vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9},
               vector<float>{1, 4, 7, 2, 5, 8, 3, 6, 9}, false, false, false);
}

NGRAPH_TEST(${BACKEND_NAME}, reshape_m2m_dim_change_transpose_plain_plain) {
  reshape_test(Shape{3, 2}, Shape{2, 3}, AxisVector{1, 0},
               vector<float>{1, 2, 3, 4, 5, 6}, vector<float>{1, 3, 5, 2, 4, 6},
               false, false, false);
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
NGRAPH_TEST(${BACKEND_NAME}, reshape_6d_plain_plain) {
  vector<float> a_data(2 * 2 * 3 * 3 * 2 * 4);
  for (int i = 0; i < 2 * 2 * 3 * 3 * 2 * 4; i++) {
    a_data[i] = float(i + 1);
  }

  reshape_test(
      Shape{2, 2, 3, 3, 2, 4}, Shape{3, 2, 2, 4, 3, 2},
      AxisVector{2, 4, 0, 5, 3, 1}, a_data,
      vector<float>{
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
          216., 288.},
      false, false, false);
}
