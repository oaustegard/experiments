#include <errno.h>
#include <stdio.h>
#include "lauxlib.h"

static int io_pclose (lua_State *L);

static int io_popen (lua_State *L) {
  const char *filename = luaL_checkstring(L, 1);
  const char *mode = luaL_optstring(L, 2, "r");
  luaL_Stream *p = (luaL_Stream *)lua_newuserdatauv(L, sizeof(luaL_Stream), 0);
  p->closef = NULL;
  luaL_setmetatable(L, LUA_FILEHANDLE);
  luaL_argcheck(L, ((mode[0] == 'r' || mode[0] == 'w') && mode[1] == '\0'), 2, "invalid mode");
  errno = 0;
  fflush(NULL);
  p->f = popen(filename, mode);
  p->closef = &io_pclose;
  return (p->f == NULL) ? luaL_fileresult(L, 0, filename) : 1;
}
