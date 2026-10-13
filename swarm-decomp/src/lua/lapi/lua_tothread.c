#include "lstate.h"
#include "lua.h"

LUA_API lua_State *lua_tothread (lua_State *L, int idx) {
  const TValue *o = index2value(L, idx);
  return (o->tt_ != 0x48) ? NULL : (lua_State *)o->value_.gc;
}
