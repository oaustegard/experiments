#include "l_noret.h"
#include "fn_luaT_objtypename.h"
l_noret luaG_typeerror (lua_State *L, const TValue *o, const char *op) {
  const char *extra = varinfo(L, o);
  const char *t = luaT_objtypename(L, o);
  luaG_runerror(L, "attempt to %s a %s value%s", op, t, extra);
}
