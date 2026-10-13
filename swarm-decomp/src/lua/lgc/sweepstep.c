#include "lstate.h"
#include "lgc_defs.h"
#include "sweepstep.h"
#include "freeobj.h"

#define GCSWEEPMAX 100

static int sweepstep (lua_State *L, global_State *g, int nextstate, GCObject **nextlist) {
  if (g->sweepgc) {
    l_mem olddebt = g->GCdebt;
    int count;
    g->sweepgc = sweeplistn(L, g->sweepgc, GCSWEEPMAX, &count);
    g->GCestimate += g->GCdebt - olddebt;
    return count;
  }
  else {
    g->gcstate = nextstate;
    g->sweepgc = nextlist;
    return 0;
  }
}
