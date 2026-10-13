#include "lobject_tags.h"
#include "fn_luaV_flttointeger.h"

static inline int LEfloatint (lua_Number f, lua_Integer i) {
  if ((lua_Unsigned)(i + ((lua_Integer)1 << 53)) <= ((lua_Unsigned)1 << 54))
    return f <= (lua_Number)i;
  else {
    lua_Integer fi;
    if (luaV_flttointeger(f, &fi, 2))
      return fi <= i;
    else
      return f < 0;
  }
}
