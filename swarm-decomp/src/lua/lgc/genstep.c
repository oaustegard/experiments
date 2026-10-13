#include "genstep.h"
#include "enterinc.h"
#include "entergen.h"
#include "setpause.h"
#include "setminordebt.h"
#include "youngcollection.h"
#include "stepgenfull.h"

static void genstep(lua_State *L, global_State *g) {
  {
    lu_mem majorbase = g->GCestimate;
    lu_mem majorinc = (majorbase / 100) * (g->genmajormul * 4);
    if (g->GCdebt > 0 && (g->totalbytes + g->GCdebt) > majorbase + majorinc) {
      lu_mem numobjs;
      enterinc(g);
      numobjs = entergen(L, g);
      if ((g->totalbytes + g->GCdebt) < majorbase + (majorinc / 2))
        return;
      g->lastatomic = numobjs;
      setpause(g);
    }
    else {
      youngcollection(L, g);
      setminordebt(g);
      g->GCestimate = majorbase;
    }
  }
}
