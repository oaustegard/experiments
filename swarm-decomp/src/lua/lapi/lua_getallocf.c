#include "lstate.h"
#include "lua.h"

#define G(L) ((L)->l_G)

LUA_API lua_Alloc lua_getallocf (lua_State *L, void **ud) {
  if (ud)
    *ud = G(L)->ud;
  return G(L)->frealloc;
}
