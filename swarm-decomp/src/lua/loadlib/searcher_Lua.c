#include "lua.h"
#include "lauxlib.h"
#include "findfile.h"

static int searcher_Lua (lua_State *L) {
  const char *filename;
  const char *name = luaL_checkstring(L, 1);
  filename = findfile(L, name, "path", "/");
  if (filename == NULL) return 1;
  if (__builtin_expect(luaL_loadfilex(L, filename, NULL) == LUA_OK, 1)) {
    lua_pushstring(L, filename);
    return 2;
  }
  else
    return luaL_error(L, "error loading module '%s' from file '%s':\n\t%s",
                      lua_tostring(L, 1), filename, lua_tostring(L, -1));
}
