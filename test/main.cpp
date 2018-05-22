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

#include <chrono>
#include <dlfcn.h>
#include <functional>
#include <iostream>

#include "gtest/gtest.h"

using namespace std;

// This is a hack that we put around since HE_SEAL and HE_HEAAN are all in libhe_backend.so
// This can be removed, if libhe_backend.so only contains one
static void register_he_backends()
{
    cout << "Registering backends" << endl;;
    void* handle = nullptr;
    string name = "libhe_backend.so";
    handle = dlopen(name.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (handle)
    {
        function<void()> create = reinterpret_cast<void (*)()>(dlsym(handle, "create_backend"));
        if (create)
        {
            create();
        }
    }
}

int main(int argc, char** argv)
{
    cout << "Registering he bakcends";
	register_he_backends();
    ::testing::InitGoogleTest(&argc, argv);
    int rc = RUN_ALL_TESTS();

    return rc;
}
