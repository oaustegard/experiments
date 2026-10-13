static void doREPL (lua_State *L) {
  int status;
  const char *oldprogname = progname;
  progname = NULL;
  while ((status = loadline(L)) != -1) {
    if (status == LUA_OK)
      status = docall(L, 0, LUA_MULTRET);
    if (status == LUA_OK) {
      int n = lua_gettop(L);
      if (n > 0) {
        luaL_checkstack(L, LUA_MINSTACK, "too many results to print");
        lua_getglobal(L, "print");
        lua_insert(L, 1);
        if (lua_pcall(L, n, 0, 0) != LUA_OK)
          l_message(progname, lua_pushfstring(L, "error calling 'print' (%s)", lua_tostring(L, -1)));
      }
    }
    else
      report(L, status);
  }
  lua_settop(L, 0);
  fwrite("\n", sizeof(char), 1, stdout);
  fflush(stdout);
  progname = oldprogname;
}
