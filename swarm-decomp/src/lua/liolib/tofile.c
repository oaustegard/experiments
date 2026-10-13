#include "lauxlib.h"

static FILE *tofile (lua_State *L) {
  luaL_Stream *p = (luaL_Stream *)luaL_checkudata(L, 1, LUA_FILEHANDLE);
  if (p->closef == NULL)
    luaL_error(L, "attempt to use a closed file");
  return p->f;
}
