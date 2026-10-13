static int codesJ (FuncState *fs, OpCode o, int sj, int k) {
  unsigned int j = sj + OFFSET_sJ;
  return luaK_code(fs, o | (j << 7) | (k << 15));
}
