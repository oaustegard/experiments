#include "lauxlib.h"

static int io_type (lua_State *L) {
  luaL_Stream *p;
  luaL_checkany(L, 1);
  p = (luaL_Stream *)luaL_testudata(L, 1, LUA_FILEHANDLE);
  if (p == NULL)
    lua_pushnil(L);
  else if (p->closef == NULL)
    lua_pushstring(L, "closed file");
  else
    lua_pushstring(L, "file");
  return 1;
}
