#include "lstate.h"
#include "fn_luaH_next.h"
#include "lua.h"

LUA_API int lua_next (lua_State *L, int idx) {
  Table *t;
  int more;
  t = (Table *)index2value(L, idx)->value_.gc;
  more = luaH_next(L, t, L->top - 1);
  if (more) {
    L->top++;
  }
  else
    L->top -= 1;
  return more;
}
