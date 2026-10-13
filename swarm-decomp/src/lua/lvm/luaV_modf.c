#include <math.h>
#include <lua.h>
#include <lstate.h>

lua_Number luaV_modf (lua_State *L, lua_Number m, lua_Number n) {
  lua_Number r;
  (void)L;
  r = fmod(m, n);
  if ((r > 0) ? n < 0 : (r < 0 && n > 0))
    r += n;
  return r;
}
