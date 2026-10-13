#include "lstate.h"
#include "Pfunc.h"
#include "fn_luaD_closeprotected.h"
#include "fn_luaD_seterrorobj.h"
#include "fn_luaD_shrinkstack.h"

int luaD_rawrunprotected (lua_State *L, Pfunc f, void *ud);

int luaD_pcall (lua_State *L, Pfunc func, void *u,
                ptrdiff_t old_top, ptrdiff_t ef) {
  int status;
  CallInfo *old_ci = L->ci;
  lu_byte old_allowhooks = L->allowhook;
  ptrdiff_t old_errfunc = L->errfunc;
  L->errfunc = ef;
  status = luaD_rawrunprotected(L, func, u);
  if (__builtin_expect(status != 0, 0)) {
    L->ci = old_ci;
    L->allowhook = old_allowhooks;
    status = luaD_closeprotected(L, old_top, status);
    luaD_seterrorobj(L, status, (StkId)((char *)L->stack + old_top));
    luaD_shrinkstack(L);
  }
  L->errfunc = old_errfunc;
  return status;
}
