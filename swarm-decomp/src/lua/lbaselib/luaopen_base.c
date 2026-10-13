#include "lua_api.h"

static const luaL_Reg base_funcs[];

int luaopen_base (lua_State *L) {
  lua_pushglobaltable(L);
  luaL_setfuncs(L, base_funcs, 0);
  lua_pushvalue(L, -1);
  lua_setfield(L, -2, "_G");
  lua_pushliteral(L, LUA_VERSION);
  lua_setfield(L, -2, "_VERSION");
  return 1;
}
