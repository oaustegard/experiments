#include <errno.h>
#include <sys/wait.h>
#include "lauxlib.h"

LUALIB_API int luaL_execresult (lua_State *L, int stat) {
  if (stat != 0 && errno != 0)
    return luaL_fileresult(L, 0, NULL);
  else {
    const char *what = "exit";
    if (WIFEXITED(stat))
      stat = WEXITSTATUS(stat);
    else if (WIFSIGNALED(stat)) {
      stat = WTERMSIG(stat);
      what = "signal";
    }
    if (*what == 'e' && stat == 0)
      lua_pushboolean(L, 1);
    else
      luaL_pushfail(L);
    lua_pushstring(L, what);
    lua_pushinteger(L, stat);
    return 3;
  }
}
