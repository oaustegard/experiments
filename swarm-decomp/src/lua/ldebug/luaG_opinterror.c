#include "l_noret.h"
#include "ltags.h"
#include "fn_luaG_typeerror.h"
l_noret luaG_opinterror (lua_State *L, const TValue *p1, const TValue *p2, const char *msg) {
  if (!ttisnumber(p1))
    p2 = p1;
  luaG_typeerror(L, p2, msg);
}
