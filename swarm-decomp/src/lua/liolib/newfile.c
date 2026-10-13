#include "newfile.h"
#include "io_fclose.h"

static inline luaL_Stream *newfile (lua_State *L) {
  luaL_Stream *p = (luaL_Stream *)lua_newuserdatauv(L, sizeof(luaL_Stream), 0);
  p->closef = NULL;
  luaL_setmetatable(L, "FILE*");
  p->f = NULL;
  return p;
}
