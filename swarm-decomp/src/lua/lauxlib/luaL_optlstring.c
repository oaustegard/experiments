#include <string.h>
#include "lauxlib.h"

const char *luaL_optlstring (lua_State *L, int arg, const char *def, size_t *len) {
  if (lua_isnoneornil(L, arg)) {
    if (len)
      *len = (def ? strlen(def) : 0);
    return def;
  }
  else {
    const char *s = lua_tolstring(L, arg, len);
    if (!s)
      luaL_typeerror(L, arg, lua_typename(L, LUA_TSTRING));
    return s;
  }
}
