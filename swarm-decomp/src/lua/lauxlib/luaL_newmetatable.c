#include "lauxlib.h"

LUALIB_API int luaL_newmetatable (lua_State *L, const char *tname) {
  if (lua_getfield(L, LUA_REGISTRYINDEX, tname) != LUA_TNIL)
    return 0;
  lua_pop(L, 1);
  lua_createtable(L, 0, 2);
  lua_pushstring(L, tname);
  lua_setfield(L, -2, "__name");
  lua_pushvalue(L, -1);
  lua_setfield(L, LUA_REGISTRYINDEX, tname);
  return 1;
}
