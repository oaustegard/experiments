#include "lstate.h"
#include "fn_luaT_gettmbyobj.h"
#include "fn_luaG_callerror.h"
#include "fn_luaC_step.h"
#include "fn_luaD_growstack.h"
#include "tryfuncTM.h"

#define l_unlikely(x) __builtin_expect(((x) != 0), 0)

#define copyobj(d, s) ((d)->value_ = (s)->value_, (d)->tt_ = (s)->tt_)

static StkId tryfuncTM (lua_State *L, StkId func) {
  const TValue *tm;
  StkId p;
  if (l_unlikely(L->stack_last - L->top <= 1)) {
    ptrdiff_t t = (char *)func - (char *)L->stack;
    if (G(L)->GCdebt > 0)
      luaC_step(L);
    luaD_growstack(L, 1, 1);
    func = (StkId)((char *)L->stack + t);
  }
  tm = luaT_gettmbyobj(L, func, 0x17);
  if (l_unlikely(((tm->tt_) & 0xf) == 0))
    luaG_callerror(L, func);
  for (p = L->top; p > func; p--)
    copyobj(p, p - 1);
  L->top++;
  copyobj(func, tm);
  return func;
}
