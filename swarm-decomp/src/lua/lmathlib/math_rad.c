static int math_rad (lua_State *L) {
  lua_pushnumber(L, luaL_checknumber(L, 1) * (3.141592653589793238462643383279502884 / 180.0));
  return 1;
}
