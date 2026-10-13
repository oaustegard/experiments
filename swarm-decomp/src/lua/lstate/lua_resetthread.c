#include "lua.h"
#include "lstate.h"
#include "fn_luaE_resetthread.h"

int lua_resetthread(lua_State *L) {
  L->nCcalls = 0;
  return luaE_resetthread(L, L->status);
}
