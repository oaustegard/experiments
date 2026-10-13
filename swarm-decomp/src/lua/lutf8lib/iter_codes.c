#include "lauxlib.h"
#include "lutf8lib_common.h"
#include "iter_auxlax.h"
#include "iter_auxstrict.h"

static int iter_codes (lua_State *L) {
  int lax = lua_toboolean(L, 2);
  const char *s = luaL_checkstring(L, 1);
  luaL_argcheck(L, !iscontp(s), 1, MSGInvalid);
  lua_pushcfunction(L, lax ? iter_auxlax : iter_auxstrict);
  lua_pushvalue(L, 1);
  lua_pushinteger(L, 0);
  return 3;
}
