#include <lua.h>
#include <lauxlib.h>

int luaopen_debug (lua_State *L) {
  luaL_checkversion(L);
  lua_createtable(L, 0, 17);
  luaL_setfuncs(L, dblib, 0);
  return 1;
}
