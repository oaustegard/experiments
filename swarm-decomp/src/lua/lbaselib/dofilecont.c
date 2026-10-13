#include "lua_api.h"

static int dofilecont (lua_State *L, int d1, lua_KContext d2) {
  (void)d1; (void)d2;
  return lua_gettop(L) - 1;
}
