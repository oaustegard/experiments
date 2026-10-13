#include "lstate.h"
#include "fn_luaD_reallocstack.h"
#include "fn_luaE_shrinkCI.h"
#include "fn_luaD_shrinkstack.h"

void luaD_shrinkstack (lua_State *L) {
  int inuse = stackinuse(L);
  int max = (inuse > LUAI_MAXSTACK / 3) ? LUAI_MAXSTACK : inuse * 3;
  if (inuse <= LUAI_MAXSTACK && (int)(L->stack_last - L->stack) > max) {
    int nsize = (inuse > LUAI_MAXSTACK / 2) ? LUAI_MAXSTACK : inuse * 2;
    luaD_reallocstack(L, nsize, 0);
  }
  luaE_shrinkCI(L);
}
