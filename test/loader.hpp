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

#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <sstream>


using namespace std;

vector<float> read_constant(const string filename)
{
    //ifstream file(filename);
    string data = file_util::read_file_to_string(filename);
    istringstream iss(data);

    vector<string> constants;
    copy(istream_iterator<string>(iss),
            istream_iterator<string>(),
            back_inserter(constants));

    vector<float> res;
    for (const string& constant : constants)
    {
        res.push_back(atof(constant.c_str()));
    }
    /* for(auto elem : res)
    {
        NGRAPH_INFO << elem << endl;
    } */

    return res;
}
