#include "lstate.h"

static int stackinuse (lua_State *L) {
  CallInfo *ci;
  int res;
  StkId lim = L->top;
  for (ci = L->ci; ci != NULL; ci = ci->previous) {
    if (lim < ci->top)
      lim = ci->top;
  }
  res = (int)(lim - L->stack) + 1;
  if (res < LUA_MINSTACK)
    res = LUA_MINSTACK;
  return res;
}
