#include "lstate.h"
#include "fn_luaT_trybinTM.h"

void luaT_trybinassocTM (lua_State *L, const TValue *p1, const TValue *p2,
                                       int flip, StkId res, TMS event) {
  if (flip) luaT_trybinTM(L, p2, p1, res, event);
  else luaT_trybinTM(L, p1, p2, res, event);
}
