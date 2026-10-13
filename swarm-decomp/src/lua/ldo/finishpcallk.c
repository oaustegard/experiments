#include "lstate.h"
#include "fn_luaF_close.h"
#include "fn_luaD_seterrorobj.h"
#include "fn_luaD_shrinkstack.h"

static int finishpcallk (lua_State *L, CallInfo *ci) {
  int status = (ci->callstatus >> 10) & 7;
  if (__builtin_expect(status == 0, 1))
    status = 1;
  else {
    StkId func = (StkId)((char *)L->stack + ci->u2.funcidx);
    L->allowhook = ci->callstatus & 1;
    func = luaF_close(L, func, status, 1);
    luaD_seterrorobj(L, status, func);
    luaD_shrinkstack(L);
    ci->callstatus &= ~0x1c00;
  }
  ci->callstatus &= ~0x10;
  L->errfunc = ci->u.c.old_errfunc;
  return status;
}
