#include "callallpendingfinalizers.h"
#include "GCTM.h"

static void callallpendingfinalizers(lua_State *L) {
  global_State *g = G(L);
  while (g->tobefnz)
    GCTM(L);
}
