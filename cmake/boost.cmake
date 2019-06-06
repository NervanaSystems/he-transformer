# ******************************************************************************
# Copyright 2018-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

include(ExternalProject)

SET(BOOST_ASIO_REPO_URL https://github.com/boostorg/asio)
SET(BOOST_SYSTEM_REPO_URL https://github.com/boostorg/system)
SET(BOOST_CONFIG_REPO_URL https://github.com/boostorg/config)
SET(BOOST_THROW_EXCEPTION_REPO_URL https://github.com/boostorg/throw_exception)
SET(BOOST_DETAIL_REPO_URL https://github.com/boostorg/detail)
SET(BOOST_ASSERT_REPO_URL https://github.com/boostorg/assert)
SET(BOOST_DATE_TIME_REPO_URL https://github.com/boostorg/date_time)
SET(BOOST_SMART_PTR_REPO_URL https://github.com/boostorg/smart_ptr)
SET(BOOST_CORE_REPO_URL https://github.com/boostorg/core)
SET(BOOST_PREDEF_REPO_URL https://github.com/boostorg/predef)
SET(BOOST_UTILITY_REPO_URL https://github.com/boostorg/utility)
SET(BOOST_TYPE_TRAITS_REPO_URL https://github.com/boostorg/type_traits)
SET(BOOST_STATIC_ASSERT_REPO_URL https://github.com/boostorg/static_assert)
SET(BOOST_MPL_REPO_URL https://github.com/boostorg/mpl)
SET(BOOST_PREPROCESSOR_REPO_URL https://github.com/boostorg/preprocessor)
SET(BOOST_NUMERIC_CONVERSION_REPO_URL https://github.com/boostorg/numeric_conversion)
SET(BOOST_BIND_REPO_URL https://github.com/boostorg/bind)
SET(BOOST_REGEX_REPO_URL https://github.com/boostorg/regex)
SET(BOOST_GIT_LABEL boost-1.69.0)

add_library(libboost INTERFACE)

ExternalProject_Add(
    ext_boost_asio
    PREFIX boost
    GIT_REPOSITORY ${BOOST_ASIO_REPO_URL}
    GIT_TAG ${BOOST_GIT_LABEL}
    # Disable install step
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    EXCLUDE_FROM_ALL TRUE
    )
ExternalProject_Get_Property(ext_boost_asio SOURCE_DIR)
message("boost asio SOURCE_DIR ${SOURCE_DIR}")
target_include_directories(libboost SYSTEM INTERFACE ${SOURCE_DIR}/include)
set(BOOST_HEADERS_PATH ${BOOST_HEADERS_PATH} ${SOURCE_DIR})

ExternalProject_Add(
    ext_boost_system
    PREFIX boost
    GIT_REPOSITORY ${BOOST_SYSTEM_REPO_URL}
    GIT_TAG ${BOOST_GIT_LABEL}
    # Disable install step
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    EXCLUDE_FROM_ALL TRUE
    BUILDE_IN_SOURCE 1
    )
ExternalProject_Get_Property(ext_boost_system SOURCE_DIR)
message("boost system SOURCE_DIR ${SOURCE_DIR}")
target_include_directories(libboost SYSTEM INTERFACE ${SOURCE_DIR}/include)
set(BOOST_HEADERS_PATH ${BOOST_HEADERS_PATH} ${SOURCE_DIR})

ExternalProject_Add(
    ext_boost_config
    PREFIX boost
    GIT_REPOSITORY ${BOOST_CONFIG_REPO_URL}
    GIT_TAG ${BOOST_GIT_LABEL}
    # Disable install step
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    EXCLUDE_FROM_ALL TRUE
    )
ExternalProject_Get_Property(ext_boost_config SOURCE_DIR)
message("boost config SOURCE_DIR ${SOURCE_DIR}")
target_include_directories(libboost SYSTEM INTERFACE ${SOURCE_DIR}/include)
set(BOOST_HEADERS_PATH ${BOOST_HEADERS_PATH} ${SOURCE_DIR})

ExternalProject_Add(
    ext_boost_throw_exception
    PREFIX boost
    GIT_REPOSITORY ${BOOST_THROW_EXCEPTION_REPO_URL}
    GIT_TAG ${BOOST_GIT_LABEL}
    # Disable install step
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    EXCLUDE_FROM_ALL TRUE
    )
ExternalProject_Get_Property(ext_boost_throw_exception SOURCE_DIR)
message("boost throw_exception SOURCE_DIR ${SOURCE_DIR}")
target_include_directories(libboost SYSTEM INTERFACE ${SOURCE_DIR}/include)
set(BOOST_HEADERS_PATH ${BOOST_HEADERS_PATH} ${SOURCE_DIR})

ExternalProject_Add(
    ext_boost_detail
    PREFIX boost
    GIT_REPOSITORY ${BOOST_DETAIL_REPO_URL}
    GIT_TAG ${BOOST_GIT_LABEL}
    # Disable install step
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    EXCLUDE_FROM_ALL TRUE
    )
ExternalProject_Get_Property(ext_boost_detail SOURCE_DIR)
message("boost detail SOURCE_DIR ${SOURCE_DIR}")
target_include_directories(libboost SYSTEM INTERFACE ${SOURCE_DIR}/include)
set(BOOST_HEADERS_PATH ${BOOST_HEADERS_PATH} ${SOURCE_DIR})

ExternalProject_Add(
    ext_boost_assert
    PREFIX boost
    GIT_REPOSITORY ${BOOST_ASSERT_REPO_URL}
    GIT_TAG ${BOOST_GIT_LABEL}
    # Disable install step
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    EXCLUDE_FROM_ALL TRUE
    )
ExternalProject_Get_Property(ext_boost_assert SOURCE_DIR)
message("boost assert SOURCE_DIR ${SOURCE_DIR}")
target_include_directories(libboost SYSTEM INTERFACE ${SOURCE_DIR}/include)
set(BOOST_HEADERS_PATH ${BOOST_HEADERS_PATH} ${SOURCE_DIR})

ExternalProject_Add(
    ext_boost_date_time
    PREFIX boost
    GIT_REPOSITORY ${BOOST_DATE_TIME_REPO_URL}
    GIT_TAG ${BOOST_GIT_LABEL}
    # Disable install step
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    EXCLUDE_FROM_ALL TRUE
    )
ExternalProject_Get_Property(ext_boost_date_time SOURCE_DIR)
message("boost date_time SOURCE_DIR ${SOURCE_DIR}")
target_include_directories(libboost SYSTEM INTERFACE ${SOURCE_DIR}/include)
set(BOOST_HEADERS_PATH ${BOOST_HEADERS_PATH} ${SOURCE_DIR})

ExternalProject_Add(
    ext_boost_smart_ptr
    PREFIX boost
    GIT_REPOSITORY ${BOOST_SMART_PTR_REPO_URL}
    GIT_TAG ${BOOST_GIT_LABEL}
    # Disable install step
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    EXCLUDE_FROM_ALL TRUE
    )
ExternalProject_Get_Property(ext_boost_smart_ptr SOURCE_DIR)
message("boost smart_ptr SOURCE_DIR ${SOURCE_DIR}")
target_include_directories(libboost SYSTEM INTERFACE ${SOURCE_DIR}/include)
set(BOOST_HEADERS_PATH ${BOOST_HEADERS_PATH} ${SOURCE_DIR})

ExternalProject_Add(
    ext_boost_core
    PREFIX boost
    GIT_REPOSITORY ${BOOST_CORE_REPO_URL}
    GIT_TAG ${BOOST_GIT_LABEL}
    # Disable install step
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    EXCLUDE_FROM_ALL TRUE
    )
ExternalProject_Get_Property(ext_boost_core SOURCE_DIR)
message("boost core SOURCE_DIR ${SOURCE_DIR}")
target_include_directories(libboost SYSTEM INTERFACE ${SOURCE_DIR}/include)
set(BOOST_HEADERS_PATH ${BOOST_HEADERS_PATH} ${SOURCE_DIR})

ExternalProject_Add(
    ext_boost_predef
    PREFIX boost
    GIT_REPOSITORY ${BOOST_PREDEF_REPO_URL}
    GIT_TAG ${BOOST_GIT_LABEL}
    # Disable install step
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    EXCLUDE_FROM_ALL TRUE
    )
ExternalProject_Get_Property(ext_boost_predef SOURCE_DIR)
message("boost predef SOURCE_DIR ${SOURCE_DIR}")
target_include_directories(libboost SYSTEM INTERFACE ${SOURCE_DIR}/include)
set(BOOST_HEADERS_PATH ${BOOST_HEADERS_PATH} ${SOURCE_DIR})

ExternalProject_Add(
    ext_boost_utility
    PREFIX boost
    GIT_REPOSITORY ${BOOST_UTILITY_REPO_URL}
    GIT_TAG ${BOOST_GIT_LABEL}
    # Disable install step
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    EXCLUDE_FROM_ALL TRUE
    )
ExternalProject_Get_Property(ext_boost_utility SOURCE_DIR)
message("boost utility SOURCE_DIR ${SOURCE_DIR}")
target_include_directories(libboost SYSTEM INTERFACE ${SOURCE_DIR}/include)
set(BOOST_HEADERS_PATH ${BOOST_HEADERS_PATH} ${SOURCE_DIR})

ExternalProject_Add(
    ext_boost_type_traits
    PREFIX boost
    GIT_REPOSITORY ${BOOST_TYPE_TRAITS_REPO_URL}
    GIT_TAG ${BOOST_GIT_LABEL}
    # Disable install step
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    EXCLUDE_FROM_ALL TRUE
    )
ExternalProject_Get_Property(ext_boost_type_traits SOURCE_DIR)
message("boost type_traits SOURCE_DIR ${SOURCE_DIR}")
target_include_directories(libboost SYSTEM INTERFACE ${SOURCE_DIR}/include)
set(BOOST_HEADERS_PATH ${BOOST_HEADERS_PATH} ${SOURCE_DIR})

ExternalProject_Add(
    ext_boost_static_assert
    PREFIX boost
    GIT_REPOSITORY ${BOOST_STATIC_ASSERT_REPO_URL}
    GIT_TAG ${BOOST_GIT_LABEL}
    # Disable install step
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    EXCLUDE_FROM_ALL TRUE
    )
ExternalProject_Get_Property(ext_boost_static_assert SOURCE_DIR)
message("boost static_assert SOURCE_DIR ${SOURCE_DIR}")
target_include_directories(libboost SYSTEM INTERFACE ${SOURCE_DIR}/include)
set(BOOST_HEADERS_PATH ${BOOST_HEADERS_PATH} ${SOURCE_DIR})

ExternalProject_Add(
    ext_boost_mpl
    PREFIX boost
    GIT_REPOSITORY ${BOOST_MPL_REPO_URL}
    GIT_TAG ${BOOST_GIT_LABEL}
    # Disable install step
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    EXCLUDE_FROM_ALL TRUE
    )
ExternalProject_Get_Property(ext_boost_mpl SOURCE_DIR)
message("boost mpl SOURCE_DIR ${SOURCE_DIR}")
target_include_directories(libboost SYSTEM INTERFACE ${SOURCE_DIR}/include)
set(BOOST_HEADERS_PATH ${BOOST_HEADERS_PATH} ${SOURCE_DIR})

ExternalProject_Add(
    ext_boost_preprocessor
    PREFIX boost
    GIT_REPOSITORY ${BOOST_PREPROCESSOR_REPO_URL}
    GIT_TAG ${BOOST_GIT_LABEL}
    # Disable install step
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    EXCLUDE_FROM_ALL TRUE
    )
ExternalProject_Get_Property(ext_boost_preprocessor SOURCE_DIR)
message("boost preprocessor SOURCE_DIR ${SOURCE_DIR}")
target_include_directories(libboost SYSTEM INTERFACE ${SOURCE_DIR}/include)
set(BOOST_HEADERS_PATH ${BOOST_HEADERS_PATH} ${SOURCE_DIR})

ExternalProject_Add(
    ext_boost_numeric_conversion
    PREFIX boost
    GIT_REPOSITORY ${BOOST_NUMERIC_CONVERSION_REPO_URL}
    GIT_TAG ${BOOST_GIT_LABEL}
    # Disable install step
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    EXCLUDE_FROM_ALL TRUE
    )
ExternalProject_Get_Property(ext_boost_numeric_conversion SOURCE_DIR)
message("boost numeric_conversion SOURCE_DIR ${SOURCE_DIR}")
target_include_directories(libboost SYSTEM INTERFACE ${SOURCE_DIR}/include)
set(BOOST_HEADERS_PATH ${BOOST_HEADERS_PATH} ${SOURCE_DIR})

ExternalProject_Add(
    ext_boost_bind
    PREFIX boost
    GIT_REPOSITORY ${BOOST_BIND_REPO_URL}
    GIT_TAG ${BOOST_GIT_LABEL}
    # Disable install step
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    EXCLUDE_FROM_ALL TRUE
    )
ExternalProject_Get_Property(ext_boost_bind SOURCE_DIR)
message("boost bind SOURCE_DIR ${SOURCE_DIR}")
target_include_directories(libboost SYSTEM INTERFACE ${SOURCE_DIR}/include)
set(BOOST_HEADERS_PATH ${BOOST_HEADERS_PATH} ${SOURCE_DIR})

ExternalProject_Add(
    ext_boost_regex
    PREFIX boost
    GIT_REPOSITORY ${BOOST_REGEX_REPO_URL}
    GIT_TAG ${BOOST_GIT_LABEL}
    # Disable install step
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    UPDATE_COMMAND ""
    EXCLUDE_FROM_ALL TRUE
    )
ExternalProject_Get_Property(ext_boost_regex SOURCE_DIR)
message("boost regex SOURCE_DIR ${SOURCE_DIR}")
target_include_directories(libboost SYSTEM INTERFACE ${SOURCE_DIR}/include)
set(BOOST_HEADERS_PATH ${BOOST_HEADERS_PATH} ${SOURCE_DIR})

message("BOOST_HEADERS_PATH ${BOOST_HEADERS_PATH}")

add_dependencies(libboost ext_boost_asio ext_boost_system ext_boost_config ext_boost_throw_exception ext_boost_detail ext_boost_assert ext_boost_date_time ext_boost_smart_ptr ext_boost_core ext_boost_predef ext_boost_utility ext_boost_type_traits ext_boost_static_assert ext_boost_mpl ext_boost_preprocessor ext_boost_numeric_conversion ext_boost_bind ext_boost_regex)
