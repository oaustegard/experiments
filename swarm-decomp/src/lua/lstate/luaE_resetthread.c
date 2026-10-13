#include "lua.h"
#include "lstate.h"
#include "fn_luaD_closeprotected.h"
#include "fn_luaD_reallocstack.h"
#include "fn_luaD_seterrorobj.h"

int luaE_resetthread (lua_State *L, int status) {
  CallInfo *ci = L->ci = &L->base_ci;
  L->stack->tt_ = 0;
  ci->func = L->stack;
  ci->callstatus = 2;
  if (status == LUA_YIELD)
    status = LUA_OK;
  L->status = LUA_OK;
  status = luaD_closeprotected(L, 1, status);
  if (status != LUA_OK)
    luaD_seterrorobj(L, status, L->stack + 1);
  else
    L->top = L->stack + 1;
  ci->top = L->top + LUA_MINSTACK;
  luaD_reallocstack(L, (int)(ci->top - L->stack), 0);
  return status;
}
