#include "lstate.h"
#include "lua.h"
#include "Closure.h"
#include "fn_luaG_getfuncline.h"
#include "funcinfo.h"
#include "getfuncname.h"
#define CIST_TAIL	(1 << 5)
#define CIST_TRAN	(1 << 8)
#define noLuaClosure(f)	((f) == NULL || (f)->c.tt != 6)
#define isLua(ci)	(!((ci)->callstatus & (1 << 1)))
#define ci_func(ci)	((LClosure *)((ci)->func->value_.gc))
#define currentpc(ci)	((int)((ci)->u.l.savedpc - ci_func(ci)->p->code) - 1)
#define getcurrentline(ci)	luaG_getfuncline(ci_func(ci)->p, currentpc(ci))

static int auxgetinfo (lua_State *L, const char *what, lua_Debug *ar, Closure *f, CallInfo *ci) {
  int status = 1;
  for (; *what; what++) {
    switch (*what) {
      case 't': {
        ar->istailcall = (ci) ? ci->callstatus & CIST_TAIL : 0;
        break;
      }
      case 'r': {
        if (ci == NULL || !(ci->callstatus & CIST_TRAN))
          ar->ftransfer = ar->ntransfer = 0;
        else {
          ar->ftransfer = ci->u2.transferinfo.ftransfer;
          ar->ntransfer = ci->u2.transferinfo.ntransfer;
        }
        break;
      }
      case 'u': {
        ar->nups = (f == NULL) ? 0 : f->c.nupvalues;
        if (noLuaClosure(f)) {
          ar->isvararg = 1;
          ar->nparams = 0;
        }
        else {
          ar->isvararg = f->l.p->is_vararg;
          ar->nparams = f->l.p->numparams;
        }
        break;
      }
      case 'S': {
        funcinfo(ar, f);
        break;
      }
      case 'n': {
        ar->namewhat = getfuncname(L, ci, &ar->name);
        if (ar->namewhat == NULL) {
          ar->namewhat = "";  /* not found */
          ar->name = NULL;
        }
        break;
      }
      case 'l': {
        ar->currentline = (ci && isLua(ci)) ? getcurrentline(ci) : -1;
        break;
      }
      case 'L':
      case 'f':  /* handled by lua_getinfo */
        break;
      default: status = 0;  /* invalid option */
    }
  }
  return status;
}
