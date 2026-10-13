static int loadline (lua_State *L) {
  int status;
  lua_settop(L, 0);
  if (!pushline(L, 1))
    return -1;
  if ((status = addreturn(L)) != LUA_OK)
    status = multiline(L);
  lua_remove(L, 1);
  return status;
}
