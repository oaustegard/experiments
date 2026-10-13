#include "lauxlib.h"

void *luaL_testudata (lua_State *L, int ud, const char *tname) {
  void *p = lua_touserdata(L, ud);
  if (p != NULL) {
    if (lua_getmetatable(L, ud)) {
      luaL_getmetatable(L, tname);
      if (!lua_rawequal(L, -1, -2))
        p = NULL;
      lua_settop(L, -3);
      return p;
    }
  }
  return NULL;
}
