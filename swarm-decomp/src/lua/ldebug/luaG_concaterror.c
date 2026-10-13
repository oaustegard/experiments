#include "l_noret.h"
#include "ltags.h"
#include "fn_luaG_typeerror.h"
#define cvt2str(o) ttisnumber(o)
l_noret luaG_concaterror (lua_State *L, const TValue *p1, const TValue *p2) {
  if (ttisstring(p1) || cvt2str(p1))
    p1 = p2;
  luaG_typeerror(L, p1, "concatenate");
}
