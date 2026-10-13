static int arith (lua_State *L, int op, const char *mtname) {
  if (tonum(L, 1) && tonum(L, 2))
    lua_arith(L, op);
  else
    trymt(L, mtname);
  return 1;
}
