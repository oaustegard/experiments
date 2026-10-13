#include "lauxlib.h"

const char *luaL_checklstring (lua_State *L, int arg, size_t *len) {
  const char *s = lua_tolstring(L, arg, len);
  if (!s) luaL_typeerror(L, arg, lua_typename(L, LUA_TSTRING));
  return s;
}
