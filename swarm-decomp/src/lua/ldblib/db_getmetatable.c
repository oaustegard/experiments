#include <lua.h>
#include <lauxlib.h>

static int db_getmetatable (lua_State *L) {
  luaL_checkany(L, 1);
  if (!lua_getmetatable(L, 1))
    lua_pushnil(L);  /* no metatable */
  return 1;
}
