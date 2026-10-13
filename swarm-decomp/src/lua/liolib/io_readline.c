#include "lauxlib.h"
#include "io_readline.h"
#include "g_read.h"
#include "aux_close.h"

static int io_readline (lua_State *L) {
  luaL_Stream *p = (luaL_Stream *)lua_touserdata(L, lua_upvalueindex(1));
  int i;
  int n = (int)lua_tointeger(L, lua_upvalueindex(2));
  if (p->closef == NULL)
    return luaL_error(L, "file is already closed");
  lua_settop(L, 1);
  luaL_checkstack(L, n, "too many arguments");
  for (i = 1; i <= n; i++)
    lua_pushvalue(L, lua_upvalueindex(3 + i));
  n = g_read(L, p->f, 2);
  if (lua_toboolean(L, -n))
    return n;
  else {
    if (n > 1)
      return luaL_error(L, "%s", lua_tostring(L, -n + 1));
    if (lua_toboolean(L, lua_upvalueindex(3))) {
      lua_settop(L, 0);
      lua_pushvalue(L, lua_upvalueindex(1));
      aux_close(L);
    }
    return 0;
  }
}
