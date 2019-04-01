from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os

__version__ = '0.0.1'


def find_he_transformer_dist_dir():
    """Return location of he-transformer library home"""

    if os.environ.get('NGRAPH_HE_BUILD_PATH'):
        ngraph_he_dist_dir = os.environ.get('NGRAPH_HE_BUILD_PATH')
    else:
        print('Must set NGRAPH_HE_BUILD_PATH')

    found = os.path.exists(os.path.join(ngraph_he_dist_dir, 'include/'))

    if not found:
        print(
            'Cannot find he-transformer library in {} make sure that '
            'NGRAPH_HE_BUILD_PATH is set correctly'.format(ngraph_he_dist_dir))
        sys.exit(1)
    else:
        print('he-transformer library found in {}'.format(ngraph_he_dist_dir))
        return ngraph_he_dist_dir


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


NGRAPH_HE_DIST_DIR = find_he_transformer_dist_dir()
NGRAPH_HE_INCLUDE_DIR = NGRAPH_HE_DIST_DIR + '/include'

ext_modules = [
    Extension(
        'python_example',
        ['he_seal_client.cpp'],
        include_dirs=[
            NGRAPH_HE_INCLUDE_DIR,
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
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
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append(
                '-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append(
                '/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)


setup(
    name='python_example',
    version=__version__,
    author='Sylvain Corlay',
    author_email='sylvain.corlay@gmail.com',
    url='https://github.com/pybind/python_example',
    description='A test project using pybind11',
    long_description='',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.2'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
