#include "lauxlib.h"

static void interror (lua_State *L, int arg) {
  if (lua_isnumber(L, arg))
    luaL_argerror(L, arg, "number has no integer representation");
  else
    luaL_typeerror(L, arg, lua_typename(L, LUA_TNUMBER));
}
