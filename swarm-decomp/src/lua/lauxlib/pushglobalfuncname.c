#include <string.h>
#include "lauxlib.h"
#include "findfield.h"

static int pushglobalfuncname (lua_State *L, lua_Debug *ar) {
  int top = lua_gettop(L);
  lua_getinfo(L, "f", ar);
  lua_getfield(L, LUA_REGISTRYINDEX, LUA_LOADED_TABLE);
  if (!lua_checkstack(L, 6))
    luaL_error(L, "stack overflow (%s)", "not enough stack");
  if (findfield(L, top + 1, 2)) {
    const char *name = lua_tolstring(L, -1, NULL);
    if (strncmp(name, LUA_GNAME ".", 3) == 0) {
      lua_pushstring(L, name + 3);
      lua_remove(L, -2);
    }
    lua_copy(L, -1, top + 1);
    lua_settop(L, top + 1);
    return 1;
  }
  else {
    lua_settop(L, top);
    return 0;
  }
}
