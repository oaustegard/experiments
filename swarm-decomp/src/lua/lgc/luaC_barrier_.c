#include "reallymarkobject.h"

void luaC_barrier_(lua_State *L, GCObject *o, GCObject *v) {
  global_State *g = L->l_G;
  if (keepinvariant(g)) {
    reallymarkobject(g, v);
    if (isold(o)) setage(v, G_OLD0);
  }
  else if (g->gckind == KGC_INC)
    makewhite(g, o);
}
