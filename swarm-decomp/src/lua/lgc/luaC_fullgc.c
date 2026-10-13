#include "lstate.h"
#include "lgc_defs.h"
#include "enterinc.h"
#include "entergen.h"
#include "fullinc.h"
#include "fn_luaC_runtilstate.h"

void luaC_fullgc (lua_State *L, int isemergency) {
  global_State *g = L->l_G;
  g->gcemergency = isemergency;
  if (g->gckind == KGC_INC)
    fullinc(L, g);
  else {
    enterinc(g);
    entergen(L, g);
  }
  g->gcemergency = 0;
}
