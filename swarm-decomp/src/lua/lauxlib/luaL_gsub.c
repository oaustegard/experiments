#include "lauxlib.h"

LUALIB_API const char *luaL_gsub (lua_State *L, const char *s, const char *p, const char *r) {
  luaL_Buffer b;
  luaL_buffinit(L, &b);
  luaL_addgsub(&b, s, p, r);
  luaL_pushresult(&b);
  return lua_tolstring(L, -1, NULL);
}
