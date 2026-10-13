#include "fn_luaO_str2num.h"
#include "l_str2dloc.h"

#define setfltvalue(obj,x) { TValue *io_ = (obj); io_->value_.n = (x); io_->tt_ = 0x13; }

static const char *l_str2int (const char *s, lua_Integer *result);
static const char *l_str2d (const char *s, lua_Number *result);

size_t luaO_str2num (const char *s, TValue *o) {
  lua_Integer i; lua_Number n;
  const char *e;
  if ((e = l_str2int(s, &i)) != NULL) {
    setivalue(o, i);
  }
  else if ((e = l_str2d(s, &n)) != NULL) {
    setfltvalue(o, n);
  }
  else
    return 0;
  return (e - s) + 1;
}
