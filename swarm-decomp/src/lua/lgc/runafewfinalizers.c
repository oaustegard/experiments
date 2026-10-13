#include "lstate.h"
#include "lgc_defs.h"

static int runafewfinalizers (lua_State *L, int n) {
  global_State *g = G(L);
  int i;
  for (i = 0; i < n && g->tobefnz; i++)
    GCTM(L);
  return i;
}
