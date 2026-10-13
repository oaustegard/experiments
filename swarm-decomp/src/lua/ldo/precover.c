#include "lstate.h"
#include "fn_luaD_rawrunprotected.h"
#include "unroll.h"

static int precover (lua_State *L, int status) {
  CallInfo *ci;
  while (status > 1 && (ci = findpcall(L)) != NULL) {
    L->ci = ci;
    ci->callstatus = (ci->callstatus & ~0x1c00) | (status << 10);
    status = luaD_rawrunprotected(L, unroll, NULL);
  }
  return status;
}
