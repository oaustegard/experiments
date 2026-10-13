#include "lua_api.h"

static int luaB_loadfile (lua_State *L) {
  const char *fname = luaL_optstring(L, 1, NULL);
  const char *mode = luaL_optstring(L, 2, NULL);
  int env = (!lua_isnone(L, 3) ? 3 : 0);
  int status = luaL_loadfilex(L, fname, mode);
  return load_aux(L, status, env);
}
