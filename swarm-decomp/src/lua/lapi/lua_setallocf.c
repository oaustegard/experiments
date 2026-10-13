#include "lstate.h"
#include "lua.h"

#define G(L) ((L)->l_G)

LUA_API void lua_setallocf (lua_State *L, lua_Alloc f, void *ud) {
  G(L)->ud = ud;
  G(L)->frealloc = f;
}
