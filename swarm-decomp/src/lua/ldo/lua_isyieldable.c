#include "lstate.h"

int lua_isyieldable (lua_State *L) {
  return (L->nCcalls & 0xffff0000) == 0;
}
