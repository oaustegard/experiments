#include "lstate.h"
#include "lobject_tags.h"
#include "fn_luaT_callorderTM.h"

int luaT_callorderiTM (lua_State *L, const TValue *p1, int v2,
                       int flip, int isf, TMS event) {
  TValue aux; const TValue *p2;
  if (isf) {
    aux.value_.n = (lua_Number)v2;
    aux.tt_ = LUA_VNUMFLT;
  }
  else {
    aux.value_.i = v2;
    aux.tt_ = LUA_VNUMINT;
  }
  if (flip) { p2 = p1; p1 = &aux; }
  else { p2 = &aux; }
  return luaT_callorderTM(L, p1, p2, event);
}
