#include "FuncState.h"
#include "OpCode.h"
#include "CREATE_ABCk.h"

int luaK_codeABCk (FuncState *fs, OpCode o, int a, int b, int c, int k) {
  return luaK_code(fs, (Instruction)a << 7 | (Instruction)b << 16 | (Instruction)k << 15 | (Instruction)o | (Instruction)c << 24);
}
