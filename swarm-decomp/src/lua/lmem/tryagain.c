#include "completestate.h"
#include "fn_luaC_fullgc.h"

static void *tryagain (lua_State *L, void *block, size_t osize, size_t nsize) {
  global_State *g = G(L);
  if (completestate(g) && !g->gcstopem) {
    luaC_fullgc(L, 1);
    return (*g->frealloc)(g->ud, block, osize, nsize);
  }
  else return NULL;
}
