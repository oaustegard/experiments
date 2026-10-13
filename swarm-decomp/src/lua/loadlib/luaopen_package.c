#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"
#include "ll_loadlib.h"
#include "ll_searchpath.h"
#include "ll_require.h"
#include "searcher_preload.h"
#include "gctm.h"
#include "setpath.h"
#include "searchpath.h"


LUAMOD_API int luaopen_package (lua_State *L) {
  static const lua_CFunction searchers[] = {
    searcher_preload, searcher_Lua, searcher_C, searcher_Croot, NULL
  };
  int i;
  luaL_getsubtable(L, LUA_REGISTRYINDEX, "_CLIBS");
  lua_createtable(L, 0, 1);
  lua_pushcclosure(L, gctm, 0);
  lua_setfield(L, -2, "__gc");
  lua_setmetatable(L, -2);
  luaL_checkversion_(L, LUA_VERSION_NUM, LUAL_NUMSIZES);
  lua_createtable(L, 0, 7);
  luaL_setfuncs(L, pk_funcs, 0);
  lua_createtable(L, 4, 0);
  for (i = 0; searchers[i] != NULL; i++) {
    lua_pushvalue(L, -2);
    lua_pushcclosure(L, searchers[i], 1);
    lua_rawseti(L, -2, i+1);
  }
  lua_setfield(L, -2, "searchers");
  setpath(L, "path", "LUA_PATH",
      "/usr/local/share/lua/5.4/?.lua;/usr/local/share/lua/5.4/?/init.lua;"
      "/usr/local/lib/lua/5.4/?.lua;/usr/local/lib/lua/5.4/?/init.lua;"
      "./?.lua;./?/init.lua");
  setpath(L, "cpath", "LUA_CPATH",
      "/usr/local/lib/lua/5.4/?.so;/usr/local/lib/lua/5.4/loadall.so;./?.so");
  lua_pushstring(L, "/\n;\n?\n!\n-\n");
  lua_setfield(L, -2, "config");
  luaL_getsubtable(L, LUA_REGISTRYINDEX, "_LOADED");
  lua_setfield(L, -2, "loaded");
  luaL_getsubtable(L, LUA_REGISTRYINDEX, "_PRELOAD");
  lua_setfield(L, -2, "preload");
  lua_rawgeti(L, LUA_REGISTRYINDEX, LUA_RIDX_GLOBALS);
  lua_pushvalue(L, -2);
  luaL_setfuncs(L, ll_funcs, 1);
  lua_settop(L, -2);
  return 1;
}
