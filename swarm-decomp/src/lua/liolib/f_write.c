#include "lauxlib.h"
#include "g_write.h"

static int f_write (lua_State *L) {
  luaL_Stream *p = (luaL_Stream *)luaL_checkudata(L, 1, LUA_FILEHANDLE);
  if (p->closef == NULL)
    luaL_error(L, "attempt to use a closed file");
  FILE *f = p->f;
  lua_pushvalue(L, 1);
  return g_write(L, f, 2);
}
