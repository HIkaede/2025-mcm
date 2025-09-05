from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
from setuptools import setup
import platform


# 根据编译器选择合适的编译参数
if platform.system() == "Windows":
    # MSVC 编译器参数
    extra_compile_args = [
        "/O2",
        "/GL",
        "/arch:AVX2",
    ]
    extra_link_args = ["/LTCG"]
else:
    # GCC/Clang 编译器参数
    extra_compile_args = [
        "-O3",
        "-march=native",
        "-ffast-math",
        "-funroll-loops",
        "-finline-functions",
    ]
    extra_link_args = []

ext_modules = [
    Pybind11Extension(
        "libsimulate",
        [
            "cpp/libsimulate.cc",
        ],
        include_dirs=[
            pybind11.get_include(),
        ],
        language="c++",
        cxx_std=17,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]


setup(
    name="libsimulate",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
    script_args=["build_ext", "--inplace"],
)
