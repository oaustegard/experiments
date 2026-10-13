#include "lgc_defs.h"
#include "fn_luaC_fix.h"

void luaC_fix(lua_State *L, GCObject *o) {
  global_State *g = G(L);
  o->marked = cast_byte((o->marked & ~63) | 4);
  g->allgc = o->next;
  o->next = g->fixedgc;
  g->fixedgc = o;
}
