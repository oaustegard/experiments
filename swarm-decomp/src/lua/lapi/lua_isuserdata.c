#include "lstate.h"
#include "lua.h"

LUA_API int lua_isuserdata (lua_State *L, int idx) {
  const TValue *o = index2value(L, idx);
  return (o->tt_ == 0x47 || o->tt_ == 2);
}
