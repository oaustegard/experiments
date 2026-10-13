#include "lauxlib.h"
#include <errno.h>

static int io_fclose (lua_State *L) {
  luaL_Stream *p = (luaL_Stream *)luaL_checkudata(L, 1, "FILE*");
  errno = 0;
  return luaL_fileresult(L, (fclose(p->f) == 0), NULL);
}
