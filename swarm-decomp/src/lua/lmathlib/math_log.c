static int math_log (lua_State *L) {
  lua_Number x = luaL_checknumber(L, 1);
  lua_Number res;
  if (lua_isnoneornil(L, 2))
    res = log(x);
  else {
    lua_Number base = luaL_checknumber(L, 2);
    if (base == 2.0)
      res = log2(x);
    else if (base == 10.0)
      res = log10(x);
    else
      res = log(x) / log(base);
  }
  lua_pushnumber(L, res);
  return 1;
}
