#include "lauxlib.h"

static int io_close (lua_State *L) {
  luaL_Stream *p;
  volatile lua_CFunction cf;
  if (lua_type(L, 1) == LUA_TNONE)
    lua_getfield(L, LUA_REGISTRYINDEX, "_IO_output");
  p = (luaL_Stream *)luaL_checkudata(L, 1, LUA_FILEHANDLE);
  if (p->closef == NULL)
    luaL_error(L, "attempt to use a closed file");
  p = (luaL_Stream *)luaL_checkudata(L, 1, LUA_FILEHANDLE);
  cf = p->closef;
  p->closef = NULL;
  return (*cf)(L);
}
