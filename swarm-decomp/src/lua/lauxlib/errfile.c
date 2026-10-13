#include <errno.h>
#include <string.h>
#include "lua.h"
#include "lauxlib.h"

static int errfile(lua_State *L, const char *what, int fnameindex) {
  int err = errno;
  const char *filename = lua_tostring(L, fnameindex) + 1;
  if (err != 0)
    lua_pushfstring(L, "cannot %s %s: %s", what, filename, strerror(err));
  else
    lua_pushfstring(L, "cannot %s %s", what, filename);
  lua_remove(L, fnameindex);
  return LUA_ERRFILE;
}
