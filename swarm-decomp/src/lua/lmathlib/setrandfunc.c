static void randseed (lua_State *L, RanState *g);

static void setrandfunc (lua_State *L) {
  RanState *state = (RanState *)lua_newuserdatauv(L, sizeof(RanState), 0);
  randseed(L, state);
  lua_settop(L, -3);
  luaL_setfuncs(L, randfuncs, 1);
}
