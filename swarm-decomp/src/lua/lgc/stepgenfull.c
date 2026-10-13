#include "stepgenfull.h"
#include "singlestep.h"
#include "atomic.h"
#include "atomic2gen.h"
#include "setminordebt.h"
#include "entersweep.h"
#include "setpause.h"

static void stepgenfull(lua_State *L, global_State *g) {
  lu_mem newatomic;
  lu_mem lastatomic = g->lastatomic;
  while (g->gcstate != GCSpropagate)
    singlestep(L);
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
