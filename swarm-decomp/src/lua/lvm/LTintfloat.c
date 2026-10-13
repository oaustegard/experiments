#include "lobject_tags.h"
#include "fn_luaV_flttointeger.h"

static inline int LTintfloat (lua_Integer i, lua_Number f) {
  if ((lua_Unsigned)(i + ((lua_Integer)1 << 53)) <= ((lua_Unsigned)1 << 54))
    return (lua_Number)i < f;
  else {
    lua_Integer fi;
    if (luaV_flttointeger(f, &fi, 2))
      return i < fi;
    else
      return f > 0;
  }
}
