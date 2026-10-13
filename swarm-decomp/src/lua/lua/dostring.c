static int dostring (lua_State *L, const char *s, const char *name) {
  int status = luaL_loadbufferx(L, s, strlen(s), name, NULL);
  if (status == LUA_OK) status = docall(L, 0, 0);
  return report(L, status);
}
