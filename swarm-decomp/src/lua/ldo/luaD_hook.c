#include "lstate.h"
#include "fn_luaD_growstack.h"

#define isLua(ci) (!((ci)->callstatus & 2))

void luaD_hook (lua_State *L, int event, int line, int ftransfer, int ntransfer) {
  lua_Hook hook = L->hook;
  if (hook && L->allowhook) {
    int mask = 8;
    CallInfo *ci = L->ci;
    ptrdiff_t top = (char *)L->top - (char *)L->stack;
    ptrdiff_t ci_top = (char *)ci->top - (char *)L->stack;
    lua_Debug ar;
    ar.event = event;
    ar.currentline = line;
    ar.i_ci = ci;
    if (ntransfer != 0) {
      mask = 0x108;
      ci->u2.transferinfo.ftransfer = ftransfer;
      ci->u2.transferinfo.ntransfer = ntransfer;
    }
    if (isLua(ci) && L->top < ci->top)
      L->top = ci->top;
    if (L->stack_last - L->top <= 20)
      luaD_growstack(L, 20, 1);
    if (ci->top < L->top + 20)
      ci->top = L->top + 20;
    L->allowhook = 0;
    ci->callstatus |= mask;
    hook(L, &ar);
    L->allowhook = 1;
    ci->top = (StkId)((char *)L->stack + ci_top);
    L->top = (StkId)((char *)L->stack + top);
    ci->callstatus &= ~mask;
  }
}
