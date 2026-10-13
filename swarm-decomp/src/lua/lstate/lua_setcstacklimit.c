#include "lua.h"
#include "lstate.h"

#define LUAI_MAXCCALLS 200

int lua_setcstacklimit(lua_State *L, unsigned int limit) {
  return LUAI_MAXCCALLS;
}
