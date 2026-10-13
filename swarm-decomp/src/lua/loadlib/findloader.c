#include <stdio.h>

#define l_unlikely(x) __builtin_expect(((x) != 0), 0)
#include "lua.h"
#include "lauxlib.h"

static void findloader (lua_State *L, const char *name) {
  lua_Integer i;
  luaL_Buffer msg;
  if (l_unlikely(lua_getfield(L, lua_upvalueindex(1), "searchers") != LUA_TTABLE))
    luaL_error(L, "'package.searchers' must be a table");
  luaL_buffinit(L, &msg);
  for (i = 1; ; i++) {
    luaL_addstring(&msg, "\n\t");
    if (l_unlikely(lua_rawgeti(L, 3, i) == LUA_TNIL)) {
      lua_settop(L, -2);
      luaL_buffsub(&msg, 2);
      luaL_pushresult(&msg);
      luaL_error(L, "module '%s' not found:%s", name, lua_tolstring(L, -1, NULL));
    }
    lua_pushstring(L, name);
    lua_callk(L, 1, 2, 0, NULL);
    if (lua_type(L, -2) == LUA_TFUNCTION)
      return;
    else if (lua_isstring(L, -2)) {
      lua_settop(L, -2);
      luaL_addvalue(&msg);
    }
    else {
      lua_settop(L, -3);
      luaL_buffsub(&msg, 2);
    }
  }
}
