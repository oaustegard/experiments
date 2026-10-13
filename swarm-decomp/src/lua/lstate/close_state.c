#include "lstate.h"
#include "fn_luaM_free_.h"
#include "fn_luaC_freeallobjects.h"
#include "fn_luaD_closeprotected.h"

static void close_state (lua_State *L) {
  global_State *g = L->l_G;
  if ((g->nilvalue.tt_ & 0xf) != 0)
    luaC_freeallobjects(L);
  else {
    L->ci = &L->base_ci;
    luaD_closeprotected(L, 1, 0);
    luaC_freeallobjects(L);
  }
  luaM_free_(L, L->l_G->strt.hash, L->l_G->strt.size * sizeof(L->l_G->strt.hash[0]));
  freestack(L);
  (*g->frealloc)(g->ud, (void *)((char *)L - 8), 0x658, 0);
}
