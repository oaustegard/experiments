#include <errno.h>
#include <stdio.h>
#include "lauxlib.h"

static int os_remove (lua_State *L) {
  const char *filename = luaL_checkstring(L, 1);
  errno = 0;
  return luaL_fileresult(L, remove(filename) == 0, filename);
}
