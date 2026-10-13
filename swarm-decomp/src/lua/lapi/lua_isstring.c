#include "lstate.h"
#include "lua.h"

LUA_API int lua_isstring (lua_State *L, int idx) {
  const TValue *o = index2value(L, idx);
  return ((o->tt_ & 0xf) == 4 || (o->tt_ & 0xf) == 3);
}
