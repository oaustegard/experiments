#include "lauxlib.h"

void luaL_checktype (lua_State *L, int arg, int t) {
  if (lua_type(L, arg) != t)
    luaL_typeerror(L, arg, lua_typename(L, t));
}
