#include "lstate.h"
#include "fn_luaD_callnoyield.h"

static void dothecall (lua_State *L, void *ud) {
  (void)ud;
  luaD_callnoyield(L, L->top - 2, 0);
}
