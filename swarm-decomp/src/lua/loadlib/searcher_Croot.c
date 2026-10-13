#include <string.h>
#include "lua.h"
#include "lauxlib.h"
#include "findfile.h"
#include "loadfunc.h"

static int searcher_Croot (lua_State *L) {
  const char *filename;
  const char *name = luaL_checkstring(L, 1);
  const char *p = strchr(name, '.');
  int stat;
  if (p == NULL) return 0;
  lua_pushlstring(L, name, p - name);
  filename = findfile(L, lua_tostring(L, -1), "cpath", "/");
  if (filename == NULL) return 1;
  if ((stat = loadfunc(L, filename, name)) != 0) {
    if (stat != 2) {
      return luaL_error(L, "error loading module '%s' from file '%s':\n\t%s",
                        lua_tostring(L, 1), filename, lua_tostring(L, -1));
    }
    else {
      lua_pushfstring(L, "no module '%s' in file '%s'", name, filename);
      return 1;
    }
  }
  lua_pushstring(L, filename);
  return 2;
}
