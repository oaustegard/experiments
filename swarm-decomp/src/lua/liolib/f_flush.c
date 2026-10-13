#include <errno.h>
#include <stdio.h>
#include "lauxlib.h"

static int f_flush (lua_State *L) {
  luaL_Stream *p = (luaL_Stream *)luaL_checkudata(L, 1, LUA_FILEHANDLE);
  FILE *f;
  if (p->closef == NULL)
    luaL_error(L, "attempt to use a closed file");
  f = p->f;
  errno = 0;
  return luaL_fileresult(L, fflush(f) == 0, NULL);
}
