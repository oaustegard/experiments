#include <lua.h>
#include <lstate.h>
#include <fn_luaG_runerror.h>

lua_Integer luaV_idiv (lua_State *L, lua_Integer m, lua_Integer n) {
  lua_Integer q;
  if ((lua_Unsigned)n + 1u <= 1u) {
    if (n == 0)
      luaG_runerror(L, "attempt to divide by zero");
    return 0 - m;
  }
  q = m / n;
  if ((m ^ n) < 0 && m % n != 0)
    q -= 1;
  return q;
}
