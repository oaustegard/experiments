#include "lauxlib.h"

lua_Number luaL_optnumber (lua_State *L, int arg, lua_Number def) {
  if (lua_isnoneornil(L, arg))
    return def;
  else {
    int isnum;
    lua_Number d = lua_tonumberx(L, arg, &isnum);
    if (__builtin_expect(!isnum, 0))
      luaL_typeerror(L, arg, lua_typename(L, LUA_TNUMBER));
    return d;
  }
}
