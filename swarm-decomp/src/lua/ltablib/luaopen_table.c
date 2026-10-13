int luaopen_table (lua_State *L) {
  luaL_checkversion(L);
  lua_createtable(L, 0, 7);
  luaL_setfuncs(L, tab_funcs, 0);
  return 1;
}
