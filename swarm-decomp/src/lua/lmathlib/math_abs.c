static int math_abs (lua_State *L) {
  if (lua_isinteger(L, 1)) {
    lua_Integer n = lua_tointegerx(L, 1, NULL);
    if (n < 0)
      n = (lua_Integer)(0u - (lua_Unsigned)n);
    lua_pushinteger(L, n);
  }
  else
    lua_pushnumber(L, fabs(luaL_checknumber(L, 1)));
  return 1;
}
