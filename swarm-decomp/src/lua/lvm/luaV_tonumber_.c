#include "lobject.h"
#include "ltags.h"
#include "l_strton.h"

int luaV_tonumber_ (const TValue *obj, lua_Number *n) {
  TValue v;
  if (ttisinteger(obj)) {
    *n = (lua_Number)ivalue(obj);
    return 1;
  }
  else if (l_strton(obj, &v)) {
    *n = ttisinteger(&v) ? (lua_Number)ivalue(&v) : fltvalue(&v);
    return 1;
  }
  else
    return 0;
}
