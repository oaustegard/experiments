int main (int argc, char **argv) {
  int status, result;
  lua_State *L = luaL_newstate();
  if (L == NULL) {
    l_message(argv[0], "cannot create state: not enough memory");
    return EXIT_FAILURE;
  }
  lua_gc(L, LUA_GCSTOP);
  lua_pushcfunction(L, &pmain);
  lua_pushinteger(L, argc);
  lua_pushlightuserdata(L, argv);
  status = lua_pcall(L, 2, 1, 0);
  result = lua_toboolean(L, -1);
  if (status != LUA_OK) {
    const char *msg = lua_tostring(L, -1);
    if (msg == NULL) msg = "(error message not a string)";
    l_message(progname, msg);
    lua_pop(L, 1);
  }
  lua_close(L);
  return (result && status == LUA_OK) ? EXIT_SUCCESS : EXIT_FAILURE;
}
