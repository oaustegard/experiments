#include <limits.h>
#include "lauxlib.h"

static int getfield (lua_State *L, const char *key, int d, int delta) {
  int isnum;
  int t = lua_getfield(L, -1, key);  /* get field and its type */
  lua_Integer res = lua_tointegerx(L, -1, &isnum);
  if (!isnum) {  /* field is not an integer? */
    if (luai_unlikely(t != LUA_TNIL))  /* some other value? */
      return luaL_error(L, "field '%s' is not an integer", key);
    else if (luai_unlikely(d == -1))  /* absent field; no default? */
      return luaL_error(L, "field '%s' missing in date table", key);
    res = d;
  }
  else {
    if (!(res >= 0 ? res - delta <= INT_MAX : INT_MIN + delta <= res))
      return luaL_error(L, "field '%s' is out-of-bound", key);
    res -= delta;
  }
  lua_settop(L, -2);  /* remove field */
  return (int)res;
}
