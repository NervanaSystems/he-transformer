include(ExternalProject)

# NTL depends on GMP
set(GMP_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/ext_gmp)
set(GMP_SOURCE_DIR ${GMP_PREFIX}/src)
message("GMP_PREFIX " ${GMP_PREFIX})
message("NGRAPH_HE_INSTALL_DIR " ${NGRAPH_HE_INSTALL_DIR})

set(GMP_INCLUDE_DIR ${GMP_SOURCE_DIR}/include)
set(GMP_LIB_DIR ${GMP_SOURCE_DIR}/lib)

ExternalProject_Add(
    ext_gmp
    DOWNLOAD_COMMAND wget https://ftp.gnu.org/gnu/gmp/gmp-6.1.2.tar.xz && tar xfJ gmp-6.1.2.tar.xz --strip 1
    PREFIX ${GMP_PREFIX}
    CONFIGURE_COMMAND
        cd ${GMP_SOURCE_DIR} && ./configure --prefix=${NGRAPH_HE_INSTALL_DIR}
    UPDATE_COMMAND ""
    BUILD_COMMAND make -j$(nproc) -C ${GMP_SOURCE_DIR}
    INSTALL_COMMAND make -j$(nproc) install -C ${GMP_SOURCE_DIR}
)

install(
    FILES
    ${NGRAPH_HE_INSTALL_INCLUDE_DIR}/gmp.h
    DESTINATION
    ${NGRAPH_INSTALL_INCLUDE_DIR}
)

install(
    FILES
    ${NGRAPH_HE_INSTALL_LIB_DIR}/libgmp.so
    DESTINATION
    ${NGRAPH_INSTALL_LIB_DIR}
)
