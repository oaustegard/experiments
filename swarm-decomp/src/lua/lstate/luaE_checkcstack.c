#include "lua.h"
#include "lstate.h"
#include "fn_luaG_runerror.h"
#include "fn_luaD_throw.h"

#define LUAI_MAXCCALLS 200

void luaE_checkcstack(lua_State *L) {
  if ((L->nCcalls & 0xffff) == LUAI_MAXCCALLS)
    luaG_runerror(L, "C stack overflow");
  else if ((L->nCcalls & 0xffff) >= (LUAI_MAXCCALLS / 10 * 11))
    luaD_throw(L, 5);
}
