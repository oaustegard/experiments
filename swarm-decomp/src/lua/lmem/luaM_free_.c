#include "fn_luaM_free_.h"

void luaM_free_ (lua_State *L, void *block, size_t osize) {
  global_State *g = G(L);
  (*g->frealloc)(g->ud, block, osize, 0);
  g->GCdebt -= osize;
}
