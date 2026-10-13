#include "lauxlib.h"
#include "aux_lines.h"
#include "opencheck.h"

static int io_lines (lua_State *L) {
  int toclose;
  luaL_Stream *p;
  if (lua_isnone(L, 1))
    lua_pushnil(L);
  if (lua_isnil(L, 1)) {
    lua_getfield(L, LUA_REGISTRYINDEX, "_IO_input");
    lua_copy(L, -1, 1);
    lua_settop(L, -2);
    p = (luaL_Stream *)luaL_checkudata(L, 1, "FILE*");
    if (p->closef == NULL)
      luaL_error(L, "attempt to use a closed file");
    toclose = 0;
  }
  else {
    const char *filename = luaL_checklstring(L, 1, NULL);
    opencheck(L, filename, "r");
    lua_copy(L, -1, 1);
    lua_settop(L, -2);
    toclose = 1;
  }
  aux_lines(L, toclose);
  if (toclose) {
    lua_pushnil(L);
    lua_pushnil(L);
    lua_pushvalue(L, 1);
    return 4;
  }
  else
    return 1;
}
