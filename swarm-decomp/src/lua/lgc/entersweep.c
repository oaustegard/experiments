static void entersweep(lua_State *L) {
  global_State *g = L->l_G;
  g->gcstate = GCSswpallgc;
  g->sweepgc = sweeptolive(L, &g->allgc);
}
