#include "lobject.h"
#include "fn_luaO_rawarith.h"
#include "fn_luaT_trybinTM.h"

void luaO_arith (lua_State *L, int op, const TValue *p1, const TValue *p2, StkId res) {
  if (!luaO_rawarith(L, op, p1, p2, res))
    luaT_trybinTM(L, p1, p2, res, op + 6);
}
