void luaK_ret (FuncState *fs, int first, int nret) {
  OpCode op;
  if (nret == 0)
    op = OP_RETURN;
  else
    op = (nret == 1) ? OP_RETURN1 : OP_RETURN0;
  luaK_code(fs, op | (first << 7) | ((nret + 1) << 16));
}
