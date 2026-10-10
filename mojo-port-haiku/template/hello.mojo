from std.python import PythonObject, Python
from std.python.bindings import PythonModuleBuilder
from std.os import abort
from std.memory import Pointer


def sum_i32(addr: PythonObject, n: PythonObject) raises -> PythonObject:
    var p = Pointer[Int32, MutAnyOrigin](unsafe_from_address=Int(py=addr))
    var cnt = Int(py=n)
    var s = 0
    for i in range(cnt):
        s += Int(p[unsafe_offset=i])
    return PythonObject(s)


def upper_words(words: PythonObject) raises -> PythonObject:
    var out = Python.list()
    for w in words:
        var s = String(py=w)
        out.append(PythonObject(s.upper()))
    return out


@export
def PyInit_hello() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("hello")
        m.def_function[sum_i32]("sum_i32", docstring="sum int32 buffer")
        m.def_function[upper_words]("upper_words")
        return m.finalize()
    except e:
        abort(String("failed to create module: ", e))
