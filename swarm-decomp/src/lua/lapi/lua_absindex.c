#include "lstate.h"
#include "lua.h"

LUA_API int lua_absindex (lua_State *L, int idx) {
  return (idx > 0 || idx <= LUA_REGISTRYINDEX)
           ? idx
           : (int)(L->top - L->ci->func) + idx;
}
