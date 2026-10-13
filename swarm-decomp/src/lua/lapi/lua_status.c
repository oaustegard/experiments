#include "lstate.h"
#include "lua.h"

LUA_API int lua_status (lua_State *L) {
  return L->status;
}
