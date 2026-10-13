#include "lauxlib.h"
#include "lualib.h"

LUAMOD_API int luaopen_os (lua_State *L) {
  luaL_newlib(L, syslib);
  return 1;
}
