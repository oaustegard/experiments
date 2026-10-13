int luaK_codeABx (FuncState *fs, OpCode o, int a, unsigned int bx) {
  return luaK_code(fs, (Instruction)o | ((Instruction)a << 7) | ((Instruction)bx << 15));
}
