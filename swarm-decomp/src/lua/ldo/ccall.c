#include "lstate.h"
#include "fn_luaD_precall.h"
#include "fn_luaD_growstack.h"
#include "fn_luaE_checkcstack.h"
#include "fn_luaV_execute.h"

static void ccall (lua_State *L, StkId func, int nResults, unsigned int inc) {
  CallInfo *ci;
  L->nCcalls += inc;
  if (__builtin_expect(((L->nCcalls & 0xffff) >= 200), 0)) {
    if (__builtin_expect((L->stack_last - L->top <= 0), 0)) {
      ptrdiff_t t = (char *)func - (char *)L->stack;
      luaD_growstack(L, 0, 1);
      func = (StkId)((char *)L->stack + t);
    }
    luaE_checkcstack(L);
  }
  if ((ci = luaD_precall(L, func, nResults)) != NULL) {
    ci->callstatus = 4;
    luaV_execute(L, ci);
  }
  L->nCcalls -= inc;
}
