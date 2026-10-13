#include "lstate.h"
#include "fn_luaD_growstack.h"

void luaD_inctop (lua_State *L) {
  if (__builtin_expect((L->stack_last - L->top <= 1), 0)) {
    luaD_growstack(L, 1, 1);
  }
  L->top++;
}
