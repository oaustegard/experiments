#include "lauxlib.h"
#include "aux_close.h"

static int f_gc (lua_State *L) {
  luaL_Stream *p = (luaL_Stream *)luaL_checkudata(L, 1, LUA_FILEHANDLE);
  if (p->closef != NULL && p->f != NULL)
    aux_close(L);
  return 0;
}
