#include "lstate.h"

void luaD_callnoyield (lua_State *L, StkId func, int nResults) {
  ccall(L, func, nResults, 0x10000 | 1);
}
