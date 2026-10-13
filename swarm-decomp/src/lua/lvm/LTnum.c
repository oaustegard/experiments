#include "lobject_tags.h"

static inline int LTnum (const TValue *l, const TValue *r) {
  if (ttisinteger(l)) {
    lua_Integer li = ivalue(l);
    if (ttisinteger(r))
      return li < ivalue(r);
    else
      return LTintfloat(li, fltvalue(r));
  }
  else {
    lua_Number lf = fltvalue(l);
    if (ttisfloat(r))
      return lf < fltvalue(r);
    else
      return LTfloatint(lf, ivalue(r));
  }
}
