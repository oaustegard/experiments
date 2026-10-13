#include "lobject_tags.h"
#include "l_strcmp.h"
#include "fn_luaT_callorderTM.h"

static inline int lessequalothers (lua_State *L, const TValue *l, const TValue *r) {
  if (ttisstring(l) && ttisstring(r))
    return l_strcmp(tsvalue(l), tsvalue(r)) <= 0;
  else
    return luaT_callorderTM(L, l, r, 21);
}
