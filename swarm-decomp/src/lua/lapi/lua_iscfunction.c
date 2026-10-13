#include "lstate.h"
#include "lua.h"

static TValue *index2value (lua_State *L, int idx);

LUA_API int lua_iscfunction (lua_State *L, int idx) {
  TValue *o = index2value(L, idx);
  return (o->tt_ == 0x16 || o->tt_ == 0x66);
}
