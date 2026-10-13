static int handle_script (lua_State *L, char **argv) {
  int status;
  const char *fname = argv[0];
  if (strcmp(fname, "-") == 0 && strcmp(argv[-1], "--") != 0)
    fname = NULL;
  status = luaL_loadfilex(L, fname, NULL);
  if (status == LUA_OK) {
    int n = pushargs(L);
    status = docall(L, n, LUA_MULTRET);
  }
  return report(L, status);
}
