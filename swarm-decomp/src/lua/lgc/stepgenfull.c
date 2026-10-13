#include "stepgenfull.h"
#include "singlestep.h"
#include "atomic.h"
#include "atomic2gen.h"
#include "setminordebt.h"
#include "entersweep.h"
#include "setpause.h"

static void stepgenfull(lua_State *L, global_State *g, lu_mem lastatomic) {
  lu_mem newatomic;
  {
    global_State *gl = G(L);
    while (gl->gcstate != GCSpropagate)
      singlestep(L);
  }
  newatomic = atomic(L);
  if (newatomic < lastatomic + (lastatomic >> 3)) {
    atomic2gen(L, g);
    setminordebt(g);
  }
  else {
    g->GCestimate = g->totalbytes + g->GCdebt;
    entersweep(L);
    while (g->gcstate != GCSpause)
      singlestep(L);
    setpause(g);
    g->lastatomic = newatomic;
  }
}
