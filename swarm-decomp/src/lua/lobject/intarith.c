#include "fn_luaV_idiv.h"
#include "fn_luaV_mod.h"
#include "fn_luaV_shiftl.h"

#define intop(op,v1,v2) ((lua_Integer)((lua_Unsigned)(v1) op (lua_Unsigned)(v2)))

static lua_Integer intarith (lua_State *L, int op, lua_Integer v1, lua_Integer v2) {
  switch (op) {
    case 0: return intop(+, v1, v2);
    case 1: return intop(-, v1, v2);
    case 2: return intop(*, v1, v2);
    case 3: return luaV_mod(L, v1, v2);
    case 6: return luaV_idiv(L, v1, v2);
    case 7: return intop(&, v1, v2);
    case 8: return intop(|, v1, v2);
    case 9: return intop(^, v1, v2);
    case 10: return luaV_shiftl(v1, v2);
    case 11: return luaV_shiftl(v1, intop(-, 0, v2));
    case 12: return intop(-, 0, v1);
    case 13: return intop(^, ~(lua_Unsigned)0, v1);
    default: return 0;
  }
}
