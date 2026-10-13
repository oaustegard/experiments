#include "lstate.h"
#include "fn_luaV_tointeger.h"
#include "lua.h"

LUA_API lua_Integer lua_tointegerx (lua_State *L, int idx, int *pisnum) {
  lua_Integer res = 0;
  const TValue *o = index2value(L, idx);
  int isnum = (__builtin_expect(o->tt_ == 3, 1))
    ? (res = o->value_.i, 1)
    : luaV_tointeger(o, &res, 0);
  if (pisnum)
    *pisnum = isnum;
  return res;
}
