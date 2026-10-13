#include "lauxlib.h"
#include "aux_lines.h"

static int f_lines (lua_State *L) {
  luaL_Stream *p = (luaL_Stream *)luaL_checkudata(L, 1, "FILE*");
  if (p->closef == NULL)
    luaL_error(L, "attempt to use a closed file");
  aux_lines(L, 0);
  return 1;
}
