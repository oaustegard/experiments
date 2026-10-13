static inline int dofile (lua_State *L, const char *name) {
  int status = luaL_loadfilex(L, name, NULL);
  if (status == LUA_OK) status = docall(L, 0, 0);
  return report(L, status);
}
