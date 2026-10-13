static int addreturn (lua_State *L) {
  const char *line = lua_tostring(L, -1);
  const char *retline = lua_pushfstring(L, "return %s;", line);
  int status = luaL_loadbufferx(L, retline, strlen(retline), "=stdin", NULL);
  if (status == LUA_OK)
    lua_remove(L, -2);
  else
    lua_pop(L, 2);
  return status;
}
