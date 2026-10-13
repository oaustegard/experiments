#include "lstate.h"
#include "lgc_defs.h"
#include "singlestep.h"
#include "propagatemark.h"
#include "atomic.h"
#include "entersweep.h"
#include "sweepstep.h"
#include "reallymarkobject.h"
#include "checkSizes.h"

#define GCFINMAX 10
#define GCFINALIZECOST 50

void luaS_resize (lua_State *L, int newsize);
static void GCTM (lua_State *L);

static lu_mem singlestep (lua_State *L) {
  global_State *g = L->l_G;
  lu_mem work;
  g->gcstopem = 1;
  switch (g->gcstate) {
    case GCSpause: {
      g->gray = g->grayagain = NULL;
      g->weak = g->allweak = g->ephemeron = NULL;
      markobject(g, g->mainthread);
      markvalue(g, &g->l_registry);
      for (int i = 0; i < 9; i++)
        markobjectN(g, g->mt[i]);
      {
        GCObject *o;
        for (o = g->tobefnz; o != NULL; o = o->next)
          markobject(g, o);
      }
      g->gcstate = GCSpropagate;
      work = 1;
      break;
    }
    case GCSpropagate: {
      if (g->gray == NULL) {
        g->gcstate = GCSenteratomic;
        work = 0;
      }
      else
        work = propagatemark(g);
      break;
    }
    case GCSenteratomic: {
      work = atomic(L);
      entersweep(L);
      g->GCestimate = g->totalbytes + g->GCdebt;
      break;
    }
    case GCSswpallgc: {
      work = sweepstep(L, g, GCSswpfinobj, &g->finobj);
      break;
    }
    case GCSswpfinobj: {
      work = sweepstep(L, g, GCSswptobefnz, &g->tobefnz);
      break;
    }
    case GCSswptobefnz: {
      work = sweepstep(L, g, GCSswpend, NULL);
      break;
    }
    case GCSswpend: {
      checkSizes(L, g);
      g->gcstate = GCScallfin;
      work = 0;
      break;
    }
    case GCScallfin: {
      if (g->tobefnz && !g->gcemergency) {
        int n;
        g->gcstopem = 0;
        n = runafewfinalizers(L, GCFINMAX);
        work = n * GCFINALIZECOST;
      }
      else {
        g->gcstate = GCSpause;
        work = 0;
      }
      break;
    }
    default:
      return 0;
  }
  g->gcstopem = 0;
  return work;
}
