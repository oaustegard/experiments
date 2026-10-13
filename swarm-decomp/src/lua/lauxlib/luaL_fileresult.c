#include <errno.h>
#include <string.h>
#include "lauxlib.h"

int luaL_fileresult (lua_State *L, int stat, const char *fname) {
  int en = errno;  /* calls to Lua API may change this value */
  if (stat) {
    lua_pushboolean(L, 1);
    return 1;
  }
  else {
    const char *msg;
    luaL_pushfail(L);
    msg = (en != 0) ? strerror(en) : "(no extra info)";
    if (fname)
      lua_pushfstring(L, "%s: %s", fname, msg);
    else
      lua_pushstring(L, msg);
    lua_pushinteger(L, en);
    return 3;
  }
}
