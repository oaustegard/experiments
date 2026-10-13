#include "lstate.h"
#include "lua.h"

#define G(L) ((L)->l_G)

LUA_API void lua_setwarnf (lua_State *L, lua_WarnFunction f, void *ud) {
  G(L)->ud_warn = ud;
  G(L)->warnf = f;
}
