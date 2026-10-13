#include "lauxlib.h"

lua_Number luaL_checknumber (lua_State *L, int arg) {
  int isnum;
  lua_Number d = lua_tonumberx(L, arg, &isnum);
  if (!isnum)
    luaL_typeerror(L, arg, lua_typename(L, LUA_TNUMBER));
  return d;
}
