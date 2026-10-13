#include "lauxlib.h"
#include "io_readline.h"
#include "aux_lines.h"

#define MAXARGLINE 250

static void aux_lines (lua_State *L, int toclose) {
  int n = lua_gettop(L) - 1;
  if (n > MAXARGLINE)
    luaL_argerror(L, MAXARGLINE + 2, "too many arguments");
  lua_pushvalue(L, 1);
  lua_pushinteger(L, n);
  lua_pushboolean(L, toclose);
  lua_rotate(L, 2, 3);
  lua_pushcclosure(L, io_readline, 3 + n);
}
