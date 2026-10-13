#include "lstate.h"
#include "lua.h"
#include "luaT_typenames_.h"

LUA_API const char *lua_typename (lua_State *L, int t) {
  (void)L;
  return luaT_typenames_[t + 1];
}
