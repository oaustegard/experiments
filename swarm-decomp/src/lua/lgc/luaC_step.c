#include "fn_luaC_step.h"
#include "genstep.h"
#include "incstep.h"
#include "stepgenfull.h"
#include "enterinc.h"

void luaC_step(lua_State *L) {
  global_State *g = G(L);
  if (g->gcstp != 0)
    luaE_setdebt(g, -2000);
  else {
    if (g->gckind == KGC_GEN) {
      if (g->lastatomic == 0) {
        genstep(L, g);
        return;
      }
      enterinc(g);
    }
    else if (g->lastatomic == 0) {
      incstep(L, g);
      return;
    }
    stepgenfull(L, g);
  }
}
