#include "lua.h"
#include "lstate.h"

int lua_gethookmask (lua_State *L) {
  return L->hookmask;
}
