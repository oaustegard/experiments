#include <stdlib.h>
#include "lauxlib.h"

static int os_getenv (lua_State *L) {
  lua_pushstring(L, getenv(luaL_checkstring(L, 1)));
  return 1;
}
