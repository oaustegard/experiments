#include "lstate.h"
#include "lobject_tags.h"
#include "lua.h"

LUA_API int lua_toboolean (lua_State *L, int idx) {
  const TValue *o = index2value(L, idx);
  return !l_isfalse(o);
}
