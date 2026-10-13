static void checktab (lua_State *L, int arg, int what) {
  if (lua_type(L, arg) != LUA_TTABLE) {
    int n = 1;
    if (lua_getmetatable(L, arg) &&
        (!(what & 1) || checkfield(L, "__index", ++n)) &&
        (!(what & 2) || checkfield(L, "__newindex", ++n)) &&
        (!(what & 4) || checkfield(L, "__len", ++n))) {
      lua_settop(L, -n-1);
    }
    else
      luaL_checktype(L, arg, LUA_TTABLE);
  }
}
