#include "ltags.h"
#include "fn_luaV_flttointeger.h"

int luaV_tointegerns (const TValue *obj, lua_Integer *p, int mode) {
  if (ttisfloat(obj))
    return luaV_flttointeger(fltvalue(obj), p, mode);
  else if (ttisinteger(obj)) {
    *p = ivalue(obj);
    return 1;
  }
  else
    return 0;
}
