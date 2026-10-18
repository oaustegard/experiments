/* Entry point for one kernel: the kernel's compiled extension is registered in
   the interpreter's builtin table (WASI has no dlopen), then CPython's normal
   main runs driver.py. -DKERNEL=<name> selects the module. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define CAT2(a, b) a##b
#define CAT(a, b) CAT2(a, b)
#define STR2(x) #x
#define STR(x) STR2(x)

extern PyObject *CAT(PyInit_, KERNEL)(void);

int main(int argc, char **argv) {
    if (PyImport_AppendInittab(STR(KERNEL), CAT(PyInit_, KERNEL)) == -1) {
        fprintf(stderr, "PyImport_AppendInittab failed\n");
        return 1;
    }
    return Py_BytesMain(argc, argv);
}
