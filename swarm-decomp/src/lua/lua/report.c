static int report (lua_State *L, int status) {
  if (status != LUA_OK) {
    const char *msg = lua_tostring(L, -1);
    if (msg == NULL) msg = "(error message not a string)";
    l_message(progname, msg);
    lua_pop(L, 1);
  }
  return status;
}
