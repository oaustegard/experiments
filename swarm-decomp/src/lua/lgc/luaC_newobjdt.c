#include "lgc_defs.h"
#include "fn_luaC_newobjdt.h"
#include "fn_luaM_malloc_.h"

GCObject *luaC_newobjdt(lua_State *L, int tt, size_t sz, size_t offset) {
  global_State *g = G(L);
  GCObject *o = (GCObject *)((char *)luaM_malloc_(L, sz, tt & 0xf) + offset);
  o->marked = luaC_white(g);
  o->tt = tt;
  o->next = g->allgc;
  g->allgc = o;
  return o;
}
