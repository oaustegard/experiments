#include "lauxlib.h"

static int aux_close (lua_State *L) {
  luaL_Stream *p = (luaL_Stream *)luaL_checkudata(L, 1, LUA_FILEHANDLE);
  volatile lua_CFunction cf = p->closef;
  p->closef = NULL;
  return (*cf)(L);
}
