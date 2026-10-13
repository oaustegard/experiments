#include "lgc_defs.h"
#include "enterinc.h"
#include "entergen.h"

void luaC_changemode(lua_State *L, int newmode) {
  global_State *g = L->l_G;
  if (newmode != g->gckind) {
    if (newmode == KGC_GEN)
      entergen(L, g);
    else
      enterinc(g);
  }
  g->lastatomic = 0;
}
