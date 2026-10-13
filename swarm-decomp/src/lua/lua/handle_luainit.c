static int handle_luainit (lua_State *L) {
  const char *name = "=LUA_INIT_5_4";
  const char *init = getenv("LUA_INIT_5_4");
  if (init == NULL) {
    name = "=LUA_INIT";
    init = getenv("LUA_INIT");
    if (init == NULL) return LUA_OK;
  }
  if (init[0] == '@')
    return dofile(L, init + 1);
  return dostring(L, init, name);
}
