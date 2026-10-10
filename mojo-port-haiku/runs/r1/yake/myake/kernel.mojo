from std.python import PythonObject, Python
from std.python.bindings import PythonModuleBuilder
from std.os import abort
from std.memory import Pointer


def sum_u32(addr: PythonObject, n: PythonObject) raises -> PythonObject:
    """Example: sum a numpy uint32 buffer given its address and length."""
    var p = Pointer[UInt32, MutAnyOrigin](unsafe_from_address=Int(py=addr))
    var total = 0
    for i in range(Int(py=n)):
        total += Int(p[unsafe_offset=i])
    return PythonObject(total)


@export
def PyInit__kernel() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_kernel")
        m.def_function[sum_u32]("sum_u32", docstring="example")
        return m.finalize()
    except e:
        abort(String("failed to create module: ", e))
