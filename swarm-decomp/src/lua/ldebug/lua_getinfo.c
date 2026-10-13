#include <string.h>
#include "lstate.h"
#include "lua.h"
#include "Closure.h"
#define ttisclosure(o)	((o)->tt_ == 0x46 || (o)->tt_ == 0x66)

LUA_API int lua_getinfo (lua_State *L, const char *what, lua_Debug *ar) {
  int status;
  Closure *cl;
  CallInfo *ci;
  TValue *func;
  if (*what == '>') {
    ci = NULL;
    func = L->top - 1;
    what++;  /* skip the '>' */
    L->top--;  /* pop function */
  }
  else {
    ci = ar->i_ci;
    func = ci->func;
  }
  cl = ttisclosure(func) ? (Closure *)func->value_.gc : NULL;
  status = auxgetinfo(L, what, ar, cl, ci);
  if (strchr(what, 'f')) {
    setobjs2s(L, L->top, func);
    L->top++;  /* api_incr_top */
  }
  if (strchr(what, 'L'))
    collectvalidlines(L, cl);
  return status;
}
