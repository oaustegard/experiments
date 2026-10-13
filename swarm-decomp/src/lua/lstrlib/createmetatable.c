static void createmetatable (lua_State *L) {
  luaL_newlibtable(L, stringmetamethods);
  luaL_setfuncs(L, stringmetamethods, 0);
  lua_pushliteral(L, "");
  lua_pushvalue(L, -2);
  lua_setmetatable(L, -2);
  lua_settop(L, -2);
  lua_pushvalue(L, -2);
  lua_setfield(L, -2, "__index");
  lua_settop(L, -2);
}
