#include "TValue.h"
#include "fn_luaV_tointegerns.h"
#include "validop.h"

static int validop (int op, TValue *v1, TValue *v2) {
  switch (op) {
    case 7: case 8: case 9: case 10: case 11: case 13: {
      lua_Integer i;
      return (luaV_tointegerns(v1, &i, 0) && luaV_tointegerns(v2, &i, 0));
    }
    case 3: case 5: case 6: {
      return (v2->tt_ == 3) ? ((lua_Number)v2->value_.i != 0) : (v2->value_.n != 0);
    }
    default: return 1;
  }
}
