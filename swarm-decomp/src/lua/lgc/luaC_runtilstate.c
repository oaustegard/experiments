#include "lgc_defs.h"
#include "singlestep.h"
#include "fn_luaC_runtilstate.h"

void luaC_runtilstate(lua_State *L, int statesmask) {
  global_State *g = G(L);
  while (!((1 << g->gcstate) & statesmask)) {
    singlestep(L);
  }
}
