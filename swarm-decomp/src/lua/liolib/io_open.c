#include <errno.h>
#include <string.h>
#include "lauxlib.h"
#include "io_fclose.h"

static int io_open (lua_State *L) {
  const char *filename = luaL_checkstring(L, 1);
  const char *mode = luaL_optstring(L, 2, "r");
  luaL_Stream *p = (luaL_Stream *)lua_newuserdatauv(L, sizeof(luaL_Stream), 0);
  const char *md = mode;
  p->closef = NULL;
  luaL_setmetatable(L, "FILE*");
  p->f = NULL;
  p->closef = io_fclose;
  luaL_argcheck(L, (*md != '\0' && strchr("rwa", *(md++)) != NULL &&
                    (*md != '+' || ((void)(++md), 1)) &&
                    (strspn(md, "b") == strlen(md))), 2, "invalid mode");
  errno = 0;
  p->f = fopen(filename, mode);
  return (p->f == NULL) ? luaL_fileresult(L, 0, filename) : 1;
}
