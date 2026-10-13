#include <errno.h>
#include <stdio.h>
#include "lauxlib.h"

static int io_flush (lua_State *L) {
  luaL_Stream *p;
  FILE *f;
  lua_getfield(L, LUA_REGISTRYINDEX, "_IO_output");
  p = (luaL_Stream *)lua_touserdata(L, -1);
  if (p->closef == NULL)
    luaL_error(L, "default %s file is closed", "output");
  f = p->f;
  errno = 0;
  return luaL_fileresult(L, fflush(f) == 0, NULL);
}
