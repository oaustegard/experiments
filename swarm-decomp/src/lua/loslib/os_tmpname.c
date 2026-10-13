#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "lauxlib.h"

static int os_tmpname (lua_State *L) {
  char buff[32];
  int err;
  strcpy(buff, "/tmp/lua_XXXXXX");
  err = mkstemp(buff);
  if (err != -1)
    close(err);
  err = (err == -1);
  if (luai_unlikely(err))
    return luaL_error(L, "unable to generate a unique filename");
  lua_pushstring(L, buff);
  return 1;
}
