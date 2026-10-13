#include "lstate.h"
#include "lgc_defs.h"
#include "Table.h"
#include "TMS.h"
#include "fn_luaT_gettm.h"

#define GCSTPCLS 4

#define gfasttm(g, et, e) \
  ((et) == NULL ? NULL : ((et)->flags & (1u << (e))) ? NULL : luaT_gettm(et, e, (g)->tmname[e]))

void luaC_checkfinalizer (lua_State *L, GCObject *o, Table *mt) {
  global_State *g = G(L);
  if ((o->marked & (1 << FINALIZEDBIT)) ||
      gfasttm(g, mt, TM_GC) == NULL ||
      (g->gcstp & GCSTPCLS))
    return;
  else {
    GCObject **p;
    if (g->gcstate >= GCSswpallgc && g->gcstate <= GCSswpend) {
      makewhite(g, o);
      if (g->sweepgc == &o->next)
        g->sweepgc = sweeptolive(L, g->sweepgc);
    }
    else
      correctpointers(g, o);
    for (p = &g->allgc; *p != o; p = &(*p)->next) {
    }
    *p = o->next;
    o->next = g->finobj;
    g->finobj = o;
    o->marked |= (1 << FINALIZEDBIT);
  }
}
