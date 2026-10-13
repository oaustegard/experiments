#include "lstate.h"
#include "lua.h"

static TValue *index2value (lua_State *L, int idx);

LUA_API int lua_isinteger (lua_State *L, int idx) {
  TValue *o = index2value(L, idx);
  return o->tt_ == 3;
}
