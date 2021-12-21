from distutils.core import setup, Extension
import sysconfig
import numpy
import sys

if sys.platform == "win32":
    extra_compile_args = ["/std:c++17"]
else:
    extra_compile_args = sysconfig.get_config_var("CFLAGS").split()
    extra_compile_args.append("-std=c++17")

module1 = Extension(
    "cpytools",
    sources=["cpytoolsmodule.cpp"],
    include_dirs=[numpy.get_include()],
    extra_compile_args=extra_compile_args,
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
                    ("PY_SSIZE_T_CLEAN", None)],
)

setup(
    name="cpytest",
    version="1.0",
    description="Python/C API Test",
    ext_modules=[module1],
)