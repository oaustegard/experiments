static void pushnumint (lua_State *L, lua_Number d) {
  if (d >= (lua_Number)LUA_MININTEGER && d < -(lua_Number)LUA_MININTEGER)
    lua_pushinteger(L, (lua_Integer)d);  /* fits in an integer */
  else
    lua_pushnumber(L, d);  /* out of range: keep it as a float */
}
