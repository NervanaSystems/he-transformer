include(ExternalProject)

set(NTL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/ext_ntl)
set(NTL_SOURCE_DIR ${NTL_PREFIX}/src/ext_ntl)

ExternalProject_Add(
    ext_ntl
    DEPENDS ext_gmp
    DOWNLOAD_COMMAND wget http://www.shoup.net/ntl/ntl-10.5.0.tar.gz
    COMMAND tar -xzf ntl-10.5.0.tar.gz -C ${NTL_SOURCE_DIR} --strip 1
    COMMAND rm ntl-10.5.0.tar.gz
    PREFIX ${NTL_PREFIX}
    CONFIGURE_COMMAND cd ${NTL_SOURCE_DIR}/src && ./configure NTL_GMP_LIP=on SHARED=on PREFIX=${NGRAPH_HE_INSTALL_DIR} GMP_PREFIX=${NGRAPH_HE_INSTALL_DIR}
    BUILD_COMMAND make -j$(nproc) -C ${NTL_SOURCE_DIR}/src
    INSTALL_COMMAND make install -C ${NTL_SOURCE_DIR}/src
)

install(
    DIRECTORY
    ${NGRAPH_HE_INSTALL_INCLUDE_DIR}/NTL/
    DESTINATION
    ${NGRAPH_INSTALL_INCLUDE_DIR}/NTL
    FILES_MATCHING PATTERN "*.h"
)

install(
    FILES
    ${NGRAPH_HE_INSTALL_LIB_DIR}/libntl.so
    DESTINATION
    ${NGRAPH_INSTALL_LIB_DIR}
)
