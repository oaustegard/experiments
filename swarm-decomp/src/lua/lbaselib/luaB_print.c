#include "lua_api.h"

static int luaB_print (lua_State *L) {
  int n = lua_gettop(L);
  int i;
  for (i = 1; i <= n; i++) {
    size_t l;
    const char *s = luaL_tolstring(L, i, &l);
    if (i > 1)
      fwrite("\t", 1, 1, stdout);
    fwrite(s, 1, l, stdout);
    lua_settop(L, -2);
  }
  fwrite("\n", 1, 1, stdout);
  fflush(stdout);
  return 0;
}
