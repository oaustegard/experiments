static int multiline (lua_State *L) {
  for (;;) {
    size_t len;
    const char *line = lua_tolstring(L, 1, &len);
    int status = luaL_loadbufferx(L, line, len, "=stdin", NULL);
    if (!incomplete(L, status) || !pushline(L, 0))
      return status;
    lua_pushstring(L, "\n");
    lua_insert(L, -2);
    lua_concat(L, 3);
  }
}
