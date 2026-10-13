#include "lauxlib.h"

static int panic (lua_State *L) {
  const char *msg = lua_type(L, -1) == LUA_TSTRING
    ? lua_tolstring(L, -1, NULL)
    : "error object is not a string";
  fprintf(stderr, "PANIC: unprotected error in call to Lua API (%s)\n", msg);
  fflush(stderr);
  return 0;
}
