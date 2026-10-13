static int sort (lua_State *L) {
  lua_Integer n = aux_getn(L, 1, TAB_RW);
  if (n > 1) {
    luaL_argcheck(L, n < INT_MAX, 1, "array too big");
    if (!lua_isnoneornil(L, 2))
      luaL_checktype(L, 2, LUA_TFUNCTION);
    lua_settop(L, 2);
    auxsort(L, 1, (IdxT)n, 0);
  }
  return 0;
}
