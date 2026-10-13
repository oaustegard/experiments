#include "lgc_defs.h"
#include "entersweep.h"
#include "singlestep.h"
#include "setpause.h"
#include "fullinc.h"
#include "fn_luaC_runtilstate.h"

static void fullinc (lua_State *L, global_State *g) {
  if (keepinvariant(g))
    entersweep(L);
  luaC_runtilstate(L, bitmask(GCSpause));
  luaC_runtilstate(L, bitmask(GCSpropagate));
  g->gcstate = GCSenteratomic;
  luaC_runtilstate(L, bitmask(GCScallfin));
  luaC_runtilstate(L, bitmask(GCSpause));
  setpause(g);
}
