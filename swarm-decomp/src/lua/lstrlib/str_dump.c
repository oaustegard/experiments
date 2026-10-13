static int str_dump (lua_State *L) {
  struct str_Writer state;
  int strip = lua_toboolean(L, 2);
  luaL_checktype(L, 1, LUA_TFUNCTION);
  lua_settop(L, 1);
  state.init = 0;
  if (lua_dump(L, writer, &state, strip) != 0)
    return luaL_error(L, "unable to dump given function");
  luaL_pushresult(&state.B);
  return 1;
}
