static void setseed (lua_State *L, Rand64 *state, lua_Unsigned n1, lua_Unsigned n2);

static void randseed (lua_State *L, RanState *g) {
  lua_Unsigned seed1 = (lua_Unsigned)time(NULL);
  lua_Unsigned seed2 = (lua_Unsigned)(size_t)L;
  setseed(L, g->s, seed1, seed2);
}
