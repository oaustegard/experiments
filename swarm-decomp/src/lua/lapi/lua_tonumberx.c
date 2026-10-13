#include "lstate.h"

#include "lua.h"

LUA_API lua_Number lua_tonumberx (lua_State *L, int idx, int *pisnum) {
  lua_Number n = 0;
  const TValue *o = index2value(L, idx);
  int isnum;
  if (o->tt_ == 0x13) {
    n = o->value_.n;
    isnum = 1;
  }
  else
    isnum = luaV_tonumber_(o, &n);
  if (pisnum) *pisnum = isnum;
  return n;
}
