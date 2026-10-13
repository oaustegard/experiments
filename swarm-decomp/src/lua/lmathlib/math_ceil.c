static void pushnumint (lua_State *L, lua_Number d);

static int math_ceil (lua_State *L) {
  if (lua_isinteger(L, 1))
    lua_settop(L, 1);
  else {
    lua_Number d = ceil(luaL_checknumber(L, 1));
    pushnumint(L, d);
  }
  return 1;
}
