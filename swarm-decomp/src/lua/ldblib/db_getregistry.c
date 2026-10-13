#include <lua.h>
#include <lauxlib.h>

static int db_getregistry (lua_State *L) {
  lua_pushvalue(L, LUA_REGISTRYINDEX);
  return 1;
}
