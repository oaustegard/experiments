static GCObject **sweeptolive_fin(lua_State *L, GCObject **p) {
  GCObject **old = p;
  do {
    p = sweeplist_fin(L, p, 1);
  } while (p == old);
  return p;
}
