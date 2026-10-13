#include <lua.h>
#include <lauxlib.h>

static int db_setcstacklimit (lua_State *L) {
  int limit = (int)luaL_checkinteger(L, 1);
  int res = lua_setcstacklimit(L, limit);
  lua_pushinteger(L, res);
  return 1;
}
