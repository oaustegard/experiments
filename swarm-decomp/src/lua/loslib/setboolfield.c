#include "lauxlib.h"

static void setboolfield (lua_State *L, const char *key, int value) {
  if (value < 0)
    return;
  lua_pushboolean(L, value);
  lua_setfield(L, -2, key);
}
