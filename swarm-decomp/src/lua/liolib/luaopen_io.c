#include "lualib.h"
#include "lauxlib.h"
#include "createstdfile.h"

LUAMOD_API int luaopen_io (lua_State *L) {
  luaL_newlib(L, iolib);
  luaL_newmetatable(L, "FILE*");
  luaL_setfuncs(L, metameth, 0);
  lua_createtable(L, 0, 7);
  luaL_setfuncs(L, meth, 0);
  lua_setfield(L, -2, "__index");
  lua_settop(L, -2);
  createstdfile(L, stdin, "_IO_input", "stdin");
  createstdfile(L, stdout, "_IO_output", "stdout");
  createstdfile(L, stderr, NULL, "stderr");
  return 1;
}
