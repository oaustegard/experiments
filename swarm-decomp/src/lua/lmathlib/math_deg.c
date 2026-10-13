static int math_deg (lua_State *L) {
  lua_pushnumber(L, luaL_checknumber(L, 1) * (180.0 / 3.141592653589793238462643383279502884));
  return 1;
}
