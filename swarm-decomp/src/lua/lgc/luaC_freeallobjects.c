#include "deletelist.h"
#include "separatetobefnz.h"
#include "callallpendingfinalizers.h"
#include "fn_luaC_changemode.h"

void luaC_freeallobjects(lua_State *L) {
  global_State *g = G(L);
  g->gcstp = 4;
  luaC_changemode(L, KGC_INC);
  separatetobefnz(g, 1);
  callallpendingfinalizers(L);
  deletelist(L, g->allgc, (GCObject *)g->mainthread);
  deletelist(L, g->fixedgc, NULL);
}
