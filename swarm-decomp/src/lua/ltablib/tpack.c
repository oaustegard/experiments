static int tpack (lua_State *L) {
  int i;
  int n = lua_gettop(L);
  lua_createtable(L, n, 1);
  lua_rotate(L, 1, 1);
  for (i = n; i >= 1; i--)
    lua_seti(L, 1, i);
  lua_pushinteger(L, n);
  lua_setfield(L, 1, "n");
  return 1;
}
