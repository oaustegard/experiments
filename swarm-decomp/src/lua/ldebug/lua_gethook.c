#include "lua.h"
#include "lstate.h"

lua_Hook lua_gethook (lua_State *L) {
  return L->hook;
}
