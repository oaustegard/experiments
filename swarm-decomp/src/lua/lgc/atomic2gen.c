#include "atomic2gen.h"
#include "cleargraylists.h"
#include "sweep2old.h"
#include "finishgencycle.h"

static void atomic2gen(lua_State *L, global_State *g) {
  cleargraylists(g);
  g->gcstate = GCSswpallgc;
  sweep2old(L, &g->allgc);
  g->firstold1 = NULL;
  g->reallyold = g->old1 = g->survival = g->allgc;
  sweep2old(L, &g->finobj);
  g->finobjrold = g->finobjold1 = g->finobjsur = g->finobj;
  sweep2old(L, &g->tobefnz);
  g->gckind = KGC_GEN;
  g->lastatomic = 0;
  g->GCestimate = (g->totalbytes + g->GCdebt);
  finishgencycle(L, g);
}
