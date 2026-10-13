#include "lobject_tags.h"

int luaV_lessthan (lua_State *L, const TValue *l, const TValue *r) {
  if (ttisnumber(l) && ttisnumber(r))
    return LTnum(l, r);
  else
    return lessthanothers(L, l, r);
}
