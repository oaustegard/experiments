#include <stdlib.h>
#include "lauxlib.h"

static int os_exit (lua_State *L) {
  int status;
  if (lua_type(L, 1) == LUA_TBOOLEAN)
    status = (lua_toboolean(L, 1) ? EXIT_SUCCESS : EXIT_FAILURE);
  else
    status = (int)luaL_optinteger(L, 1, EXIT_SUCCESS);
  if (lua_toboolean(L, 2))
    lua_close(L);
  if (L) exit(status);
  return 0;
}
