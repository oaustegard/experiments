#include <errno.h>
#include <stdio.h>
#include "lauxlib.h"

static int io_pclose (lua_State *L) {
  luaL_Stream *p = (luaL_Stream *)luaL_checkudata(L, 1, LUA_FILEHANDLE);
  errno = 0;
  return luaL_execresult(L, pclose(p->f));
}
