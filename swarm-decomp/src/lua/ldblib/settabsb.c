#include <lua.h>
#include <lauxlib.h>

static void settabsb (lua_State *L, const char *k, int v) {
  lua_pushboolean(L, v);
  lua_setfield(L, -2, k);
}
