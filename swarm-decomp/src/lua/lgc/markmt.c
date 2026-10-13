static void markmt(global_State *g) {
  int i;
  for (i = 0; i < LUA_NUMTYPES; i++)
    markobjectN(g, g->mt[i]);
}
