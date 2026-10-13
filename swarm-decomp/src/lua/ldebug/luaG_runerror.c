#include <stdarg.h>
#include "lstate.h"
#include "LClosure.h"
#include "fn_luaC_step.h"
#include "fn_luaO_pushvfstring.h"
#include "fn_luaG_addinfo.h"
#include "fn_luaG_getfuncline.h"
#include "fn_luaG_errormsg.h"
#include "fn_luaG_runerror.h"

#define isLua(ci)	(!((ci)->callstatus & (1 << 1)))
#define ci_func(ci)	((LClosure *)((ci)->func->value_.gc))
#define currentpc(ci)	((int)((ci)->u.l.savedpc - ci_func(ci)->p->code) - 1)

_Noreturn void luaG_runerror (lua_State *L, const char *fmt, ...) {
  CallInfo *ci = L->ci;
  const char *msg;
  va_list argp;
  if (G(L)->GCdebt > 0)  /* error message uses memory */
    luaC_step(L);
  va_start(argp, fmt);
  msg = luaO_pushvfstring(L, fmt, argp);  /* format message */
  va_end(argp);
  if (isLua(ci)) {  /* if Lua function, add source:line information */
    luaG_addinfo(L, msg, ci_func(ci)->p->source,
                 luaG_getfuncline(ci_func(ci)->p, currentpc(ci)));
    setobjs2s(L, L->top - 2, L->top - 1);  /* remove 'msg' */
    L->top--;
  }
  luaG_errormsg(L);
}
