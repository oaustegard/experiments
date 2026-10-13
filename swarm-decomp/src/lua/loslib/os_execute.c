#include <errno.h>
#include <stdlib.h>
#include "lauxlib.h"

static int os_execute (lua_State *L) {
  const char *cmd = luaL_optlstring(L, 1, NULL, NULL);
  int stat;
  errno = 0;
  stat = system(cmd);
  if (cmd != NULL)
    return luaL_execresult(L, stat);
  else {
    lua_pushboolean(L, stat);
    return 1;
  }
}
