#include "lstate.h"
#include "lgc_defs.h"
#include "singlestep.h"
#include "atomic.h"
#include "atomic2gen.h"
#include "fn_luaE_setdebt.h"

static lu_mem entergen (lua_State *L, global_State *g) {
  lu_mem numobjs;
  {
    global_State *gl = G(L);
    while (gl->gcstate != GCSpause)
      singlestep(L);
  }
  {
    global_State *gl = G(L);
    while (gl->gcstate != GCSpropagate)
      singlestep(L);
  }
  numobjs = atomic(L);
  atomic2gen(L, g);
  luaE_setdebt(g, -(((((lu_mem)g->totalbytes + g->GCdebt)) / 100) * g->genminormul));
  return numobjs;
}
