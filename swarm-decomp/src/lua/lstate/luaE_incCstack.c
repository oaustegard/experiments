#include "lstate.h"
#include "fn_luaG_runerror.h"
#include "fn_luaD_throw.h"

void luaE_incCstack (lua_State *L) {
  L->nCcalls++;
  if ((L->nCcalls & 0xffff) >= 200)
    luaE_checkcstack(L);
}
