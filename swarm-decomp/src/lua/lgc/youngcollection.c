#include "youngcollection.h"
#include "markold.h"
#include "sweepgen.h"
#include "atomic.h"
#include "finishgencycle.h"

static void youngcollection(lua_State *L, global_State *g) {
  GCObject **psurvival;
  GCObject *dummy;
  if (g->firstold1) {
    markold(g, g->firstold1, g->reallyold);
    g->firstold1 = NULL;
  }
  markold(g, g->finobj, g->finobjrold);
  markold(g, g->tobefnz, NULL);
  atomic(L);
  g->gcstate = GCSswpallgc;
  psurvival = sweepgen(L, g, &g->allgc, g->survival, &g->firstold1);
  sweepgen(L, g, psurvival, g->old1, &g->firstold1);
  g->reallyold = g->old1;
  g->old1 = *psurvival;
  g->survival = g->allgc;
  dummy = NULL;
  psurvival = sweepgen(L, g, &g->finobj, g->finobjsur, &dummy);
  sweepgen(L, g, psurvival, g->finobjold1, &dummy);
  g->finobjrold = g->finobjold1;
  g->finobjold1 = *psurvival;
  g->finobjsur = g->finobj;
  sweepgen(L, g, &g->tobefnz, NULL, &dummy);
  finishgencycle(L, g);
}
