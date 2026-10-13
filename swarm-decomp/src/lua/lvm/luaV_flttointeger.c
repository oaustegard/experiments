#include <math.h>
#include "fn_luaV_flttointeger.h"

int luaV_flttointeger (lua_Number n, lua_Integer *p, int mode) {
  lua_Number f = floor(n);
  if (n != f) {
    if (mode == 0)
      return 0;
    else if (mode == 2)
      f += 1;
  }
  return (f >= -9223372036854775808.0 && f < 9223372036854775808.0 &&
          (*(p) = (lua_Integer)(f), 1));
}
