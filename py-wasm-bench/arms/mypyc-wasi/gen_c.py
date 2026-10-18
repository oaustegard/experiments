"""Emit C for one kernel with mypyc. Does not invoke a host C compiler.

usage: gen_c.py KERNEL_SRC BUILD_DIR
Copies the kernel unmodified into BUILD_DIR (so the module is named after the
kernel), runs mypycify with target_dir=BUILD_DIR/gen, and prints the absolute path
of the generated translation unit (gen/__native.c) as the last stdout line. The
lib-rt C sources are #included by that file; build.sh adds mypyc.build.include_dir()
to the include path.
"""
import os
import shutil
import sys

# Target is wasm32 (ILP32), the build host is x86_64. mypyc derives its platform
# size from sysconfig SIZEOF_SIZE_T at import time and otherwise emits 8-byte pointer
# strides for list literals (lower/list_ops.py uses PLATFORM_SIZE), which corrupts
# every list literal on wasm32. Pin the target size before mypyc is imported.
import sysconfig

_real_get_config_var = sysconfig.get_config_var


def _target_config_var(name, *args, **kwargs):
    if name == "SIZEOF_SIZE_T":
        return 4
    return _real_get_config_var(name, *args, **kwargs)


sysconfig.get_config_var = _target_config_var

from mypyc.build import mypycify

src, bdir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
name = os.path.splitext(os.path.basename(src))[0]
os.makedirs(bdir, exist_ok=True)
shutil.copyfile(src, os.path.join(bdir, name + ".py"))
os.chdir(bdir)
exts = mypycify([name + ".py"], target_dir="gen", opt_level="2")
assert len(exts) == 1, exts
srcs = [os.path.join(bdir, s) for s in exts[0].sources]
print("\n".join(srcs))
