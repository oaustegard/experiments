#include "lstate.h"

void luaD_call (lua_State *L, StkId func, int nResults) {
  ccall(L, func, nResults, 1);
}
