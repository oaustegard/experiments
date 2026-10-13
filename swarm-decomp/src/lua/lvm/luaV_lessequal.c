#include "lobject_tags.h"

int luaV_lessequal (lua_State *L, const TValue *l, const TValue *r) {
  if (ttisnumber(l) && ttisnumber(r))
    return LEnum(l, r);
  else
    return lessequalothers(L, l, r);
}
