void *luaM_realloc_ (lua_State *L, void *block, size_t osize, size_t nsize) {
  void *newblock;
  global_State *g = G(L);
  newblock = (*g->frealloc)(g->ud, block, osize, nsize);
  if (newblock == NULL && nsize > 0) {
    newblock = tryagain(L, block, osize, nsize);
    if (newblock == NULL) return NULL;
  }
  g->GCdebt = (g->GCdebt + nsize) - osize;
  return newblock;
}
