static int pmain (lua_State *L) {
  int argc = (int)lua_tointeger(L, 1);
  char **argv = (char **)lua_touserdata(L, 2);
  int script;
  int args = collectargs(argv, &script);
  int optlim = (script > 0) ? script : argc;
  luaL_checkversion(L);
  if (args == has_error) {
    print_usage(argv[script]);
    return 0;
  }
  if (args & has_v) print_version();
  if (args & has_E) {
    lua_pushboolean(L, 1);
    lua_setfield(L, LUA_REGISTRYINDEX, "LUA_NOENV");
  }
  luaL_openlibs(L);
  createargtable(L, argv, argc, script);
  lua_gc(L, LUA_GCRESTART);
  lua_gc(L, LUA_GCGEN, 0, 0);
  if (!(args & has_E)) {
    if (handle_luainit(L) != LUA_OK) return 0;
  }
  if (!runargs(L, argv, optlim)) return 0;
  if (script > 0) {
    if (handle_script(L, argv + script) != LUA_OK) return 0;
  }
  if (args & has_i) doREPL(L);
  else if (script < 1 && !(args & (has_e | has_v))) {
    if (isatty(0)) {
      print_version();
      doREPL(L);
    }
    else dofile(L, NULL);
  }
  lua_pushboolean(L, 1);
  return 1;
}
