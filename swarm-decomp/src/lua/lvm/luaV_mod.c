#include <lua.h>
#include <lstate.h>
#include <fn_luaG_runerror.h>

lua_Integer luaV_mod (lua_State *L, lua_Integer m, lua_Integer n) {
  lua_Integer r;
  if (__builtin_expect((lua_Unsigned)n + 1u <= 1u, 0)) {
    if (n == 0)
      luaG_runerror(L, "attempt to perform 'n%%0'");
    return 0;
  }
  r = m % n;
  if (r != 0 && (r ^ n) < 0)
    r += n;
  return r;
}
