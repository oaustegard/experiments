#include "lua_api.h"

static int pairscont (lua_State *L, int status, lua_KContext k) {
  (void)L; (void)status; (void)k;
  return 3;
}
