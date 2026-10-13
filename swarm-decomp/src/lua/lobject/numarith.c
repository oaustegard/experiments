#include <math.h>
#include "fn_luaV_modf.h"

static lua_Number numarith (lua_State *L, int op, lua_Number v1, lua_Number v2) {
  switch (op) {
    case 0: return v1 + v2;
    case 1: return v1 - v2;
    case 2: return v1 * v2;
    case 5: return v1 / v2;
    case 4: return (v2 == 2) ? v1 * v1 : pow(v1, v2);
    case 6: return floor(v1 / v2);
    case 12: return -v1;
    case 3: return luaV_modf(L, v1, v2);
    default: return 0;
  }
}
