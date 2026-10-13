#include "lua.h"
#include "lstate.h"

int lua_gethookcount (lua_State *L) {
  return L->basehookcount;
}
