#include "lauxlib.h"
#include "g_read.h"

static int io_read (lua_State *L) {
  luaL_Stream *p;
  lua_getfield(L, LUA_REGISTRYINDEX, "_IO_input");
  p = (luaL_Stream *)lua_touserdata(L, -1);
  if (p->closef == NULL)
    luaL_error(L, "default %s file is closed", "input");
  return g_read(L, p->f, 1);
}
