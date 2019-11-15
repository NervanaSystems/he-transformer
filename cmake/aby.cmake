# ******************************************************************************
# Copyright 2018-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
# ******************************************************************************

include(ExternalProject)


set(ABY_REPO_URL https://github.com/encryptogroup/ABY.git)
set(ABY_GIT_TAG public)
set(ABY_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/ext_aby)
set(ABY_SRC_DIR ${ABY_PREFIX}/src/)
set(ABY_INCLUDE_DIR ${ABY_PREFIX}/include)
set(ABY_PATCH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/aby_perf_stats.patch)

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
  add_compile_options(-Wno-vla-extension)
endif()

message(STATUS "ABY_PATCH_COMMAND ${ABY_PATCH_COMMAND}")
ExternalProject_Add(ext_aby
                    PREFIX ${ABY_PREFIX}
                    GIT_REPOSITORY ${ABY_REPO_URL}
                    GIT_TAG ${ABY_GIT_TAG}
                    CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                               -DBOOST_ROOT=${BOOST_HEADERS_PATH}
                               -DCMAKE_INSTALL_PREFIX=${ABY_PREFIX}
                               -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
                               -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                    DEPENDS libboost
                    INSTALL_DIR ${EXTERNAL_INSTALL_DIR}
                    PATCH_COMMAND ${ABY_PATCH_COMMAND}
                    UPDATE_COMMAND "")

ExternalProject_Get_Property(ext_aby SOURCE_DIR BINARY_DIR)

# ABY requires GMP / OpenSSL
find_package(GMP REQUIRED)
find_package(GMPXX REQUIRED)
find_package(OpenSSL REQUIRED)

set(ABY_LIB_DIR ${BINARY_DIR}/lib})

message("ABY_INCLUDE_DIR ${ABY_INCLUDE_DIR}")

if(NOT IS_DIRECTORY ${ABY_INCLUDE_DIR})
  file(MAKE_DIRECTORY ${ABY_INCLUDE_DIR})
endif()

if(NOT IS_DIRECTORY ${ABY_LIB_DIR})
  file(MAKE_DIRECTORY ${ABY_LIB_DIR})
endif()

add_library(libaby_orig STATIC IMPORTED)
set(ABY_LIB "${BINARY_DIR}/lib/libaby.a")
set_target_properties(libaby_orig
                      PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                 "${ABY_INCLUDE_DIR}"
                                 IMPORTED_LOCATION
                                 ${ABY_LIB})
add_dependencies(libaby_orig ext_aby)
target_link_libraries(libaby_orig
                      INTERFACE GMP::GMPXX
                                GMP::GMP
                                OpenSSL::SSL
                                OpenSSL::Crypto)

# For filesystem
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(Apple)?Clang$")
  target_link_libraries(libaby_orig INTERFACE c++experimental stdc++fs)
else() # GCC
  target_link_libraries(libaby_orig INTERFACE stdc++fs)
endif()

add_library(librelic_s STATIC IMPORTED)
set(RELIC_LIB "${BINARY_DIR}/lib/librelic_s.a")
set_target_properties(librelic_s
                      PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                 "${ABY_INCLUDE_DIR}"
                                 IMPORTED_LOCATION
                                 ${RELIC_LIB})
add_dependencies(librelic_s ext_aby)

add_library(libencrypto_utils STATIC IMPORTED)
set(ENCRYPTO_UTILS_LIB "${BINARY_DIR}/lib/libencrypto_utils.a")
set_target_properties(libencrypto_utils
                      PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                 "${ABY_INCLUDE_DIR}"
                                 IMPORTED_LOCATION
                                 ${ENCRYPTO_UTILS_LIB})
add_dependencies(libencrypto_utils ext_aby)

add_library(libotextension STATIC IMPORTED)
set(OTEXTENSION_LIB "${BINARY_DIR}/lib/libotextension.a")
set_target_properties(libotextension
                      PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
                                 "${ABY_INCLUDE_DIR}"
                                 IMPORTED_LOCATION
                                 ${OTEXTENSION_LIB})
add_dependencies(libotextension ext_aby)

add_library(libaby INTERFACE)
target_link_libraries(libaby
                      INTERFACE libaby_orig
                                libotextension
                                libencrypto_utils
                                librelic_s
                                GMP::GMPXX
                                GMP::GMP
                                OpenSSL::SSL
                                OpenSSL::Crypto)

# Create symbolic links in external install dir TODO: just install there
# directly
file(MAKE_DIRECTORY ${EXTERNAL_INSTALL_LIB_DIR})

add_custom_target(libaby_install ALL
                  DEPENDS ext_aby
                  COMMAND ${CMAKE_COMMAND}
                          -E
                          create_symlink
                          ${BINARY_DIR}/lib/libotextension.a
                          ${EXTERNAL_INSTALL_LIB_DIR}/libotextension.a
                  COMMAND ${CMAKE_COMMAND}
                          -E
                          create_symlink
                          ${BINARY_DIR}/lib/libencrypto_utils.a
                          ${EXTERNAL_INSTALL_LIB_DIR}/libencrypto_utils.a
                  COMMAND ${CMAKE_COMMAND}
                          -E
                          create_symlink
                          ${BINARY_DIR}/lib/librelic_s.a
                          ${EXTERNAL_INSTALL_LIB_DIR}/librelic_s.a
                  COMMAND ${CMAKE_COMMAND}
                          -E
                          create_symlink
                          ${BINARY_DIR}/lib/libaby.a
                          ${EXTERNAL_INSTALL_LIB_DIR}/libaby.a)

install(DIRECTORY ${ABY_INCLUDE_DIR}/
        DESTINATION ${EXTERNAL_INSTALL_INCLUDE_DIR}
        FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.h")
