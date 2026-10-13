#include "lstate.h"
#include "lua.h"

#define G(L) ((L)->l_G)

LUA_API lua_CFunction lua_atpanic (lua_State *L, lua_CFunction panicf) {
  lua_CFunction old;
  old = G(L)->panic;
  G(L)->panic = panicf;
  return old;
}
