#include "io_noclose.h"

static int io_noclose (lua_State *L) {
  luaL_Stream *p = (luaL_Stream *)luaL_checkudata(L, 1, "FILE*");
  p->closef = &io_noclose;
  lua_pushnil(L);
  lua_pushstring(L, "cannot close standard file");
  return 2;
}
