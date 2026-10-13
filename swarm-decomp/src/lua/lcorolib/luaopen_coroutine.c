#include "lua_api.h"

int luaopen_coroutine (lua_State *L) {
  luaL_checkversion(L);
  lua_createtable(L, 0, 8);
  luaL_setfuncs(L, co_funcs, 0);
  return 1;
}
