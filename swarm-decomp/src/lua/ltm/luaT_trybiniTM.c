#include "lstate.h"
#include "lobject_tags.h"
#include "fn_luaT_trybinassocTM.h"

void luaT_trybiniTM (lua_State *L, const TValue *p1, lua_Integer i2,
                                   int flip, StkId res, TMS event) {
  TValue aux;
  aux.value_.i = i2;
  aux.tt_ = LUA_VNUMINT;
  luaT_trybinassocTM(L, p1, &aux, flip, res, event);
}
