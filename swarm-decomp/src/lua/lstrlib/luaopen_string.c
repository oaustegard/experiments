int luaopen_string (lua_State *L) {
  luaL_checkversion(L);
  luaL_newlibtable(L, strlib);
  luaL_setfuncs(L, strlib, 0);
  luaL_newlibtable(L, stringmetamethods);
  luaL_setfuncs(L, stringmetamethods, 0);
  lua_pushliteral(L, "");
  lua_pushvalue(L, -2);
  lua_setmetatable(L, -2);
  lua_settop(L, -2);
  lua_pushvalue(L, -2);
  lua_setfield(L, -2, "__index");
  lua_settop(L, -2);
  return 1;
}
