#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>

extern const luaL_Reg loadedlibs[];

void luaL_openlibs (lua_State *L) {
  const luaL_Reg *lib;
  for (lib = loadedlibs; lib->func; lib++) {
    luaL_requiref(L, lib->name, lib->func, 1);
    lua_settop(L, -2);
  }
}
