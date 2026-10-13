#include "lstate.h"
#include "lua.h"

LUA_API int lua_gettop (lua_State *L) {
  return (int)(L->top - (L->ci->func + 1));
}
