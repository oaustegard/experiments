static int tunpack (lua_State *L) {
  lua_Unsigned n;
  lua_Integer i = luaL_optinteger(L, 2, 1);
  lua_Integer e = lua_isnoneornil(L, 3) ? luaL_len(L, 1) : luaL_checkinteger(L, 3);
  if (i > e) return 0;
  n = (lua_Unsigned)e - i;
  if (n >= (unsigned int)INT_MAX) goto err;
  if (!lua_checkstack(L, (int)(++n))) goto err;
  for (; i < e; i++) {
    lua_geti(L, 1, i);
  }
  lua_geti(L, 1, e);
  return (int)n;
err:
  return luaL_error(L, "too many results to unpack");
}
