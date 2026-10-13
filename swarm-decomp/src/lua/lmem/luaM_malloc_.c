#include "fn_luaD_throw.h"
#include "fn_luaM_realloc_.h"

void *luaM_malloc_ (lua_State *L, size_t size, int tag) {
  if (size == 0) return NULL;
  else {
    global_State *g = G(L);
    void *newblock = (*g->frealloc)(g->ud, NULL, tag, size);
    if (__builtin_expect(newblock == NULL, 0)) {
      newblock = tryagain(L, NULL, tag, size);
      if (newblock == NULL) luaD_throw(L, LUA_ERRMEM);
    }
    g->GCdebt += size;
    return newblock;
  }
}
