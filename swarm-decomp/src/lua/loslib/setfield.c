#include "lauxlib.h"

static void setfield (lua_State *L, const char *key, int value, int delta) {
  lua_pushinteger(L, (lua_Integer)value + delta);
  lua_setfield(L, -2, key);
}
