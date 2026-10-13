#include <stddef.h>
#include "lstate.h"
#include "fn_luaD_call.h"
#include "fn_luaD_pcall.h"
#include "CallS.h"
#include "lua.h"

static StkId index2stack (lua_State *L, int idx);
static void f_call (lua_State *L, void *ud);

#define CIST_OAH (1 << 0)
#define CIST_YPCALL (1 << 4)
#define yieldable(L) ((((L)->nCcalls) & 0xffff0000) == 0)
#define savestack(L,p) ((char *)(p) - (char *)(L)->stack)

LUA_API int lua_pcallk (lua_State *L, int nargs, int nresults, int errfunc,
                        lua_KContext ctx, lua_KFunction k) {
  struct CallS c;
  int status;
  ptrdiff_t func;
  if (errfunc == 0)
    func = 0;
  else {
    StkId o = index2stack(L, errfunc);
    func = savestack(L, o);
  }
  c.func = L->top - (nargs + 1);
  if (k == NULL || !yieldable(L)) {
    c.nresults = nresults;
    status = luaD_pcall(L, f_call, &c, savestack(L, c.func), func);
  }
  else {
    CallInfo *ci = L->ci;
    ci->u.c.k = k;
    ci->u.c.ctx = ctx;
    ci->u2.funcidx = (int)savestack(L, c.func);
    ci->u.c.old_errfunc = L->errfunc;
    L->errfunc = func;
    ci->callstatus = (ci->callstatus & ~CIST_OAH) | L->allowhook;
    ci->callstatus |= CIST_YPCALL;
    luaD_call(L, c.func, nresults);
    ci->callstatus &= ~CIST_YPCALL;
    L->errfunc = ci->u.c.old_errfunc;
    status = 0;
  }
  if (nresults <= -1 && L->ci->top < L->top)
    L->ci->top = L->top;
  return status;
}
