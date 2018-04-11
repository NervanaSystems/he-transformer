include(ExternalProject)

set(NTL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/external_ntl)
set(NTL_SOURCE_DIR ${NTL_PREFIX}/src/external_ntl)

ExternalProject_Add(
    ext_ntl
    DEPENDS ext_gmp
    DOWNLOAD_COMMAND wget http://www.shoup.net/ntl/ntl-10.5.0.tar.gz
    COMMAND tar -xzf ntl-10.5.0.tar.gz -C ${NTL_SOURCE_DIR} --strip 1
    COMMAND rm ntl-10.5.0.tar.gz
    PREFIX ${NTL_PREFIX}
    CONFIGURE_COMMAND cd ${NTL_SOURCE_DIR}/src && ./configure NTL_THREADS=on NTL_THREAD_BOOST=on NTL_EXCEPTIONS=on SHARED=on PREFIX=${PYHE_INSTALL_DIR}
    BUILD_COMMAND make -j$(nproc) -C ${NTL_SOURCE_DIR}/src
    INSTALL_COMMAND make install -C ${NTL_SOURCE_DIR}/src
    BUILD_ALWAYS 1
)
