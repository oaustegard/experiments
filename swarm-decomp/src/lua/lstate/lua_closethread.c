#include "lua.h"
#include "lstate.h"
#include "fn_luaE_resetthread.h"

int lua_closethread(lua_State *L, lua_State *from) {
  L->nCcalls = (from) ? (from->nCcalls & 0xffff) : 0;
  return luaE_resetthread(L, L->status);
}
