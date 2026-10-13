#include "lua.h"
#include "lstate.h"
#include "lobject_setters.h"
#include "fn_luaG_findlocal.h"

const char *lua_setlocal (lua_State *L, const lua_Debug *ar, int n) {
  StkId pos = NULL;  /* to avoid warnings */
  const char *name;
  name = luaG_findlocal(L, ar->i_ci, n, &pos);
  if (name) {
    setobjs2s(L, pos, L->top - 1);
    L->top--;  /* pop value */
  }
  return name;
}
