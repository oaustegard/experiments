#include "lstate.h"
#include "unroll.h"
#include "fn_luaD_rawrunprotected.h"
#include "fn_luaD_seterrorobj.h"
#include "fn_luaS_new.h"

#define l_unlikely(x) __builtin_expect(((x) != 0), 0)

static void resume (lua_State *L, void *ud);

int lua_resume (lua_State *L, lua_State *from, int nargs, int *nresults) {
  int status;
  if (L->status == 0) {
    if (L->ci != &L->base_ci)
      return resume_error(L, "cannot resume non-suspended coroutine", nargs);
    else if (L->top - (L->ci->func + 1) == nargs)
      return resume_error(L, "cannot resume dead coroutine", nargs);
  }
  else if (L->status != 1)
    return resume_error(L, "cannot resume dead coroutine", nargs);
  L->nCcalls = (from) ? (from->nCcalls & 0xffff) : 0;
  if ((L->nCcalls & 0xffff) >= 200)
    return resume_error(L, "C stack overflow", nargs);
  L->nCcalls++;
  status = luaD_rawrunprotected(L, resume, &nargs);
  status = precover(L, status);
  if (l_unlikely(status > 1)) {
    L->status = (lu_byte)status;
    luaD_seterrorobj(L, status, L->top);
    L->ci->top = L->top;
  }
  *nresults = (status == 1) ? L->ci->u2.nyield
                            : (int)(L->top - (L->ci->func + 1));
  return status;
}
