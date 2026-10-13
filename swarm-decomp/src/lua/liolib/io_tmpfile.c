#include "lauxlib.h"
#include <errno.h>
#include <stdio.h>

static int io_fclose (lua_State *L);

static int io_tmpfile (lua_State *L) {
  luaL_Stream *p = (luaL_Stream *)lua_newuserdatauv(L, sizeof(luaL_Stream), 0);
  p->closef = NULL;
  luaL_setmetatable(L, "FILE*");
  p->f = NULL;
  p->closef = &io_fclose;
  errno = 0;
  p->f = tmpfile();
  return (p->f == NULL) ? luaL_fileresult(L, 0, NULL) : 1;
}
