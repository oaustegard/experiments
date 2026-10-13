#include "lgc_defs.h"
#include "fn_luaC_newobj.h"
#include "fn_luaM_malloc_.h"

GCObject *luaC_newobj(lua_State *L, int tt, size_t sz) {
  global_State *g = G(L);
  GCObject *o = (GCObject *)luaM_malloc_(L, sz, tt & 0xf);
  o->marked = luaC_white(g);
  o->tt = tt;
  o->next = g->allgc;
  g->allgc = o;
  return o;
}
