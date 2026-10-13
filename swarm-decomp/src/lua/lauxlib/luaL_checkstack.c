#include "lauxlib.h"

void luaL_checkstack (lua_State *L, int space, const char *msg) {
  if (__builtin_expect(!lua_checkstack(L, space), 0)) {
    if (msg)
      luaL_error(L, "stack overflow (%s)", msg);
    else
      luaL_error(L, "stack overflow");
  }
}
