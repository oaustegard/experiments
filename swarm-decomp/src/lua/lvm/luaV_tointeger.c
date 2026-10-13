#include "lobject.h"
#include <math.h>
#include "lstate.h"
#include "ltags.h"
#include "Table.h"
#include "TString.h"
#include "fn_luaO_str2num.h"
#include "l_strton.h"

#define lua_numbertointeger(n,p) \
  ((n) >= (lua_Number)(-9223372036854775807LL - 1) && \
   (n) < -(lua_Number)(-9223372036854775807LL - 1) && \
   (*(p) = (lua_Integer)(n), 1))

static int flttointns (lua_Number n, lua_Integer *p, int mode) {
  lua_Number f = floor(n);
  if (n != f) {
    if (mode == 0) return 0;
    else if (mode == 2) f += 1;
  }
  return lua_numbertointeger(f, p);
}

int luaV_tointeger (const TValue *obj, lua_Integer *p, int mode) {
  TValue v;
  if (l_strton(obj, &v))
    obj = &v;
  if (ttisfloat(obj))
    return flttointns(fltvalue(obj), p, mode);
  else {
    if (ttisinteger(obj)) {
      *p = ivalue(obj);
      return 1;
    }
    return 0;
  }
}
