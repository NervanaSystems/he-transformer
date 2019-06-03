from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os

# TODO: get from environment
__version__ = '0.0.0-dev'

PYNGRAPH_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
BOOST_ROOT_DIR = os.path.abspath(os.path.dirname(__file__))


def find_he_transformer_dist_dir():
    """Return location of he-transformer library home"""

    if os.environ.get('NGRAPH_HE_BUILD_PATH'):
        ngraph_he_dist_dir = os.environ.get('NGRAPH_HE_BUILD_PATH')
    else:
        print('Must set NGRAPH_HE_BUILD_PATH')
        sys.exit(1)

    found = os.path.exists(os.path.join(ngraph_he_dist_dir, 'include'))

    if not found:
        print(
            'Cannot find he-transformer library in {} make sure that '
            'NGRAPH_HE_BUILD_PATH is set correctly'.format(ngraph_he_dist_dir))
        sys.exit(1)
    else:
        print('he-transformer library found in {}'.format(ngraph_he_dist_dir))
        return ngraph_he_dist_dir


def find_pybind_headers_dir():
    """Return location of pybind11 headers."""
    if os.environ.get('PYBIND_HEADERS_PATH'):
        pybind_headers_dir = os.environ.get('PYBIND_HEADERS_PATH')
    else:
        pybind_headers_dir = os.path.join(PYNGRAPH_ROOT_DIR, 'pybind11')

    found = os.path.exists(
        os.path.join(pybind_headers_dir, 'include/pybind11'))
    if not found:
        print(
            'Cannot find pybind11 library in {} make sure that '
            'PYBIND_HEADERS_PATH is set correctly'.format(pybind_headers_dir))
        sys.exit(1)
    else:
        print('pybind11 library found in {}'.format(pybind_headers_dir))
        return pybind_headers_dir


def find_boost_headers_dir():
    """Return location of boost headers."""
    if os.environ.get('BOOST_HEADERS_PATH'):
        boost_headers_dirs = os.environ.get('BOOST_HEADERS_PATH')
        boost_headers_dirs = boost_headers_dirs.split(';')
        boost_headers_dirs = [x + '/include' for x in boost_headers_dirs]
    else:
        boost_headers_dirs = [os.path.join(BOOST_ROOT_DIR)]

    found_asio = any([
        os.path.exists(os.path.join(header_dir, 'boost/asio'))
        for header_dir in boost_headers_dirs
    ])
    found_system = any([
        os.path.exists(os.path.join(header_dir, 'boost/system'))
        for header_dir in boost_headers_dirs
    ])
    found_config = any([
        os.path.exists(os.path.join(header_dir, 'boost/config'))
        for header_dir in boost_headers_dirs
    ])
    if not found_asio:
        print('Cannot find boost asio library in {} make sure that '
              'BOOST_HEADERS_PATH is set correctly'.format(boost_headers_dirs))
        sys.exit(1)
    if not found_system:
        print('Cannot find boost system library in {} make sure that '
              'BOOST_HEADERS_PATH is set correctly'.format(boost_headers_dirs))
        sys.exit(1)
    if not found_config:
        print('Cannot find boost config library in {} make sure that '
              'BOOST_HEADERS_PATH is set correctly'.format(boost_headers_dirs))
        sys.exit(1)
    print('boost library found in {}'.format(boost_headers_dirs))
    return boost_headers_dirs


def find_cxx_compiler():
    """Returns C++ compiler."""
    if os.environ.get('CXX_COMPILER'):
        print('CXX_COMPILER', os.environ.get('CXX_COMPILER'))
        return os.environ.get('CXX_COMPILER')
    else:
        return 'CXX'


def find_c_compiler():
    """Returns C compiler."""
    if os.environ.get('C_COMPILER'):
        print('C_COMPILER', os.environ.get('C_COMPILER'))
        return os.environ.get('C_COMPILER')
    else:
        return 'CC'


os.environ["CXX"] = find_cxx_compiler()
os.environ["CC"] = find_c_compiler()

PYBIND11_INCLUDE_DIR = find_pybind_headers_dir() + '/include'
NGRAPH_HE_DIST_DIR = find_he_transformer_dist_dir()
NGRAPH_HE_INCLUDE_DIR = os.path.join(NGRAPH_HE_DIST_DIR, 'include')
NGRAPH_HE_LIB_DIR = os.path.join(NGRAPH_HE_DIST_DIR, 'lib')
BOOST_INCLUDE_DIRS = find_boost_headers_dir()

print('NGRAPH_HE_DIST_DIR', NGRAPH_HE_DIST_DIR)
print('NGRAPH_HE_LIB_DIR ', NGRAPH_HE_LIB_DIR)
print('NGRAPH_HE_INCLUDE_DIR', NGRAPH_HE_INCLUDE_DIR)
print('BOOST_INCLUDE_DIRS', BOOST_INCLUDE_DIRS)

include_dirs = [PYNGRAPH_ROOT_DIR, NGRAPH_HE_INCLUDE_DIR, PYBIND11_INCLUDE_DIR]
include_dirs.extend(BOOST_INCLUDE_DIRS)

library_dirs = [NGRAPH_HE_LIB_DIR]

libraries = ['he_seal_client']

data_files = [('lib', [(NGRAPH_HE_LIB_DIR + '/' + library)
                       for library in os.listdir(NGRAPH_HE_LIB_DIR)])]

sources = ['py_he_seal_client/he_seal_client.cpp']

sources = [PYNGRAPH_ROOT_DIR + '/' + source for source in sources]

print('sources', sources)

ext_modules = [
    Extension(
        'he_seal_client',
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        language='c++'),
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.
    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++1z'):
        return '-std=c++1z'
    elif has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')


def add_platform_specific_link_args(link_args):
    """Add linker flags specific for actual OS."""
    if sys.platform.startswith('linux'):
        link_args += ['-Wl,-rpath,$ORIGIN/../..']
        link_args += ['-z', 'noexecstack']
        link_args += ['-z', 'relro']
        link_args += ['-z', 'now']


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    def _add_extra_compile_arg(self, flag, compile_args):
        """Return True if successfully added given flag to compiler args."""
        if has_flag(self.compiler, flag):
            compile_args += [flag]
            return True
        return False

    def build_extensions(self):
        """Build extension providing extra compiler flags."""
        # -Wstrict-prototypes is not a valid option for c++
        try:
            self.compiler.compiler_so.remove('-Wstrict-prototypes')
        except (AttributeError, ValueError):
            pass
        for ext in self.extensions:
            ext.extra_compile_args += [cpp_flag(self.compiler)]

            if not self._add_extra_compile_arg('-fstack-protector-strong',
                                               ext.extra_compile_args):
                self._add_extra_compile_arg('-fstack-protector',
                                            ext.extra_compile_args)

            self._add_extra_compile_arg('-fvisibility=hidden',
                                        ext.extra_compile_args)
            self._add_extra_compile_arg('-flto', ext.extra_compile_args)
            self._add_extra_compile_arg('-fPIC', ext.extra_compile_args)
            add_platform_specific_link_args(ext.extra_link_args)

            ext.extra_compile_args += ['-Wformat', '-Wformat-security']
            ext.extra_compile_args += ['-O2', '-D_FORTIFY_SOURCE=2']
        build_ext.build_extensions(self)


setup(
    name='he_seal_client',
    version=__version__,
    author='Intel Corporation',
    url='https://github.com/NervanaSystems/he-transformer',
    description='Client for HE-transformer',
    long_description='',
    ext_modules=ext_modules,
    data_files=data_files,
    install_requires=['pybind11>=2.2'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False)
