#include "lua.h"
#include "lauxlib.h"
#include "findfile.h"
#include "loadfunc.h"

static int searcher_C (lua_State *L) {
  const char *name = luaL_checkstring(L, 1);
  const char *filename = findfile(L, name, "cpath", "/");
  if (filename == NULL) return 1;
  if (__builtin_expect(loadfunc(L, filename, name) == 0, 1)) {
    lua_pushstring(L, filename);
    return 2;
  }
  else
    return luaL_error(L, "error loading module '%s' from file '%s':\n\t%s",
                      lua_tostring(L, 1), filename, lua_tostring(L, -1));
}
