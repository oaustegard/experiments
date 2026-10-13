#include "lstate.h"
#include "fn_luaF_close.h"
#include "lua.h"

LUA_API void lua_settop (lua_State *L, int idx) {
  CallInfo *ci;
  StkId func, newtop;
  ptrdiff_t diff;
  ci = L->ci;
  func = ci->func;
  if (idx >= 0) {
    diff = ((func + 1) + idx) - L->top;
    for (; diff > 0; diff--)
      L->top++->tt_ = 0;
  }
  else {
    diff = idx + 1;
  }
  newtop = L->top + diff;
  if (diff < 0 && L->tbclist >= newtop) {
    newtop = luaF_close(L, newtop, -1, 0);
  }
  L->top = newtop;
}
