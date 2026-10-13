#include "lauxlib.h"

void *luaL_checkudata (lua_State *L, int ud, const char *tname) {
  void *p = luaL_testudata(L, ud, tname);
  if (p == NULL) luaL_typeerror(L, ud, tname);
  return p;
}
