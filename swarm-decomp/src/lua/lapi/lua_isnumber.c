#include "lstate.h"
#include "lua.h"

static TValue *index2value (lua_State *L, int idx);
int luaV_tonumber_ (const TValue *obj, lua_Number *n);

LUA_API int lua_isnumber (lua_State *L, int idx) {
  lua_Number n;
  const TValue *o = index2value(L, idx);
  return (o->tt_ == 0x13) ? (n = o->value_.n, 1) : luaV_tonumber_(o, &n);
}
