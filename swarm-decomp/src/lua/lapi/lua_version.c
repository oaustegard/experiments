#include "lua.h"

LUA_API lua_Number lua_version (lua_State *L) {
  (void)L;
  return LUA_VERSION_NUM;
}
