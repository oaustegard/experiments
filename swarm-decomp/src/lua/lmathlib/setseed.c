static Rand64 nextrand (Rand64 *state);

static void setseed (lua_State *L, Rand64 *state, lua_Unsigned n1, lua_Unsigned n2) {
  int i;
  state[0] = n1;
  state[1] = 0xff;
  state[2] = n2;
  state[3] = 0;
  for (i = 0; i < 16; i++)
    nextrand(state);
  lua_pushinteger(L, n1);
  lua_pushinteger(L, n2);
}
