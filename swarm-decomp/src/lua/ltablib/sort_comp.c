static int sort_comp (lua_State *L, int a, int b) {
  if (lua_type(L, 2) == LUA_TNIL)
    return lua_compare(L, a, b, LUA_OPLT);
  else {
    int res;
    lua_pushvalue(L, 2);
    lua_pushvalue(L, a-1);
    lua_pushvalue(L, b-2);
    lua_callk(L, 2, 1, 0, NULL);
    res = lua_toboolean(L, -1);
    lua_settop(L, -2);
    return res;
  }
}
