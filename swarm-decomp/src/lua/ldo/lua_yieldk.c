#include "lstate.h"
#include "fn_luaG_runerror.h"
#include "fn_luaD_throw.h"

int lua_yieldk (lua_State *L, int nresults, lua_KContext ctx, lua_KFunction k) {
  CallInfo *ci;
  ci = L->ci;
  if (__builtin_expect(!((L->nCcalls & 0xffff0000) == 0), 0)) {
    if (L != L->l_G->mainthread)
      luaG_runerror(L, "attempt to yield across a C-call boundary");
    else
      luaG_runerror(L, "attempt to yield from outside a coroutine");
  }
  L->status = 1;
  ci->u2.nyield = nresults;
  if (ci->callstatus & (1 << 1)) {
    if ((ci->u.c.k = k) != NULL)
      ci->u.c.ctx = ctx;
    luaD_throw(L, 1);
  }
  return 0;
}
