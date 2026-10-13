void luaH_free(lua_State *L, Table *t) {
  if (!(t->lastfree == NULL))
    luaM_free_(L, t->node, (size_t)(1 << t->lsizenode) * sizeof(Node));
  luaM_free_(L, t->array, (size_t)luaH_realasize(t) * sizeof(TValue));
  luaM_free_(L, t, sizeof(Table));
}
