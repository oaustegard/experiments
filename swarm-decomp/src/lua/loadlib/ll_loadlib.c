#include "lauxlib.h"
#include "ll_loadlib.h"
#include "lookforfunc.h"

static int ll_loadlib (lua_State *L) {
  const char *path = luaL_checklstring(L, 1, NULL);
  const char *init = luaL_checklstring(L, 2, NULL);
  int stat = lookforfunc(L, path, init);
  if (stat) {
    lua_pushnil(L);
    lua_rotate(L, -2, 1);
    lua_pushstring(L, (stat == ERRLIB) ? "open" : "init");
    return 3;
  }
  return 1;
}
