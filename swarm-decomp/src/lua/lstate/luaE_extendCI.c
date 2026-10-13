#include "lua.h"
#include "lstate.h"
#include "fn_luaM_malloc_.h"

CallInfo *luaE_extendCI(lua_State *L) {
  CallInfo *ci;
  ci = (CallInfo *)luaM_malloc_(L, sizeof(CallInfo), 0);
  L->ci->next = ci;
  ci->previous = L->ci;
  ci->next = NULL;
  ci->u.l.trap = 0;
  L->nci++;
  return ci;
}
