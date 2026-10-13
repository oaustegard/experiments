#include "lua.h"
#include "lauxlib.h"
#include "findloader.h"

static int ll_require (lua_State *L) {
  const char *name = luaL_checklstring(L, 1, NULL);
  lua_settop(L, 1);
  lua_getfield(L, LUA_REGISTRYINDEX, "_LOADED");
  lua_getfield(L, 2, name);
  if (lua_toboolean(L, -1))
    return 1;
  lua_settop(L, -2);
  findloader(L, name);
  lua_rotate(L, -2, 1);
  lua_pushvalue(L, 1);
  lua_pushvalue(L, -3);
  lua_callk(L, 2, 1, 0, NULL);
  if (lua_type(L, -1) != LUA_TNIL)
    lua_setfield(L, 2, name);
  else
    lua_settop(L, -2);
  if (lua_getfield(L, 2, name) == LUA_TNIL) {
    lua_pushboolean(L, 1);
    lua_copy(L, -1, -2);
    lua_setfield(L, 2, name);
  }
  lua_rotate(L, -2, 1);
  return 2;
}
