#include "l_noret.h"
#include "fn_luaV_tointegerns.h"
#include "fn_luaG_runerror.h"

const char *varinfo (lua_State *L, const TValue *o);

l_noret luaG_tointerror (lua_State *L, const TValue *p1, const TValue *p2) {
  lua_Integer temp;
  if (!luaV_tointegerns(p1, &temp, 0))
    p2 = p1;
  luaG_runerror(L, "number%s has no integer representation", varinfo(L, p2));
}
