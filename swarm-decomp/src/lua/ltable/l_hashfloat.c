#include <math.h>
#include <limits.h>
#include "lobject.h"

static int l_hashfloat (lua_Number n) {
  int i;
  lua_Integer ni;
  n = frexp(n, &i) * -(lua_Number)INT_MIN;
  if (!(n >= (lua_Number)(LUA_MININTEGER) && n < -(lua_Number)(LUA_MININTEGER)))
    return 0;
  else {
    unsigned int u;
    ni = (lua_Integer)n;
    u = (unsigned int)i + (unsigned int)ni;
    return (int)(u <= (unsigned int)INT_MAX ? u : ~u);
  }
}
