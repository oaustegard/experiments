#include "lstate.h"
#include "fn_luaG_errors.h"

void luaT_tryconcatTM (lua_State *L) {
  StkId top = L->top;
  if (!callbinTM(L, top-2, top-1, top-2, TM_CONCAT))
    luaG_concaterror(L, top-2, top-1);
}
