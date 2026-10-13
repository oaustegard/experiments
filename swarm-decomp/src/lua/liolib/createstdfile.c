#include "createstdfile.h"
#include "io_noclose.h"

static inline void createstdfile (lua_State *L, FILE *f, const char *k, const char *fname) {
  luaL_Stream *p = (luaL_Stream *)lua_newuserdatauv(L, sizeof(luaL_Stream), 0);
  p->closef = NULL;
  luaL_setmetatable(L, "FILE*");
  p->f = f;
  p->closef = &io_noclose;
  if (k != NULL) {
    lua_pushvalue(L, -1);
    lua_setfield(L, LUA_REGISTRYINDEX, k);
  }
  lua_setfield(L, -2, fname);
}
