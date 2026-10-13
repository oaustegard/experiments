#include "lauxlib.h"
#include "g_write.h"

static int io_write (lua_State *L) {
  luaL_Stream *p;
  lua_getfield(L, LUA_REGISTRYINDEX, "_IO_output");
  p = (luaL_Stream *)lua_touserdata(L, -1);
  if (p->closef == NULL)
    luaL_error(L, "default %s file is closed", "output");
  return g_write(L, p->f, 1);
}
