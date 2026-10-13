#include "l_noret.h"
#include "fn_luaD_call.h"
#include "fn_luaD_throw.h"

#define restorestack(L,n)	((StkId)((char *)(L)->stack + (n)))

l_noret luaG_errormsg (lua_State *L) {
  if (L->errfunc != 0) {  /* is there an error handling function? */
    StkId errfunc = restorestack(L, L->errfunc);
    setobjs2s(L, L->top, L->top - 1);  /* move argument */
    setobjs2s(L, L->top - 1, errfunc);  /* push function */
    L->top++;  /* assume EXTRA_STACK */
    luaD_callnoyield(L, L->top - 2, 1);  /* call it */
  }
  luaD_throw(L, LUA_ERRRUN);
}
