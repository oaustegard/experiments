#include "lauxlib.h"
#include "ll_searchpath.h"
#include "searchpath.h"

static int ll_searchpath (lua_State *L) {
  const char *dirsep = luaL_optlstring(L, 4, "/", NULL);
  const char *sep = luaL_optlstring(L, 3, ".", NULL);
  const char *path = luaL_checklstring(L, 2, NULL);
  const char *name = luaL_checklstring(L, 1, NULL);
  const char *f = searchpath(L, name, path, sep, dirsep);
  if (f != NULL)
    return 1;
  lua_pushnil(L);
  lua_rotate(L, -2, 1);
  return 2;
}
