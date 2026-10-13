#include "lua.h"
#include "lstate.h"
#include "LClosure.h"
#include "ltags.h"
#include "lobject_setters.h"
#include "fn_luaF_getlocalname.h"
#include "fn_luaG_findlocal.h"

#define isLfunction(o)	((o)->tt_ == (LUA_VLCL | (1 << 6)))

const char *lua_getlocal (lua_State *L, const lua_Debug *ar, int n) {
  const char *name;
  if (ar == NULL) {  /* information about non-active function? */
    if (!isLfunction(L->top - 1))  /* not a Lua function? */
      name = NULL;
    else  /* consider live variables at function start (parameters) */
      name = luaF_getlocalname(((LClosure *)((L->top - 1)->value_.gc))->p, n, 0);
  }
  else {  /* active function; get information through 'ar' */
    StkId pos = NULL;  /* to avoid warnings */
    name = luaG_findlocal(L, ar->i_ci, n, &pos);
    if (name) {
      setobjs2s(L, L->top, pos);
      L->top++;
    }
  }
  return name;
}
