#include "lstate.h"
#include "lua.h"

LUA_API int lua_type (lua_State *L, int idx) {
  const TValue *o = index2value(L, idx);
  return ((((o->tt_ & 0xf) != 0) || o != &L->l_G->nilvalue) ? (o->tt_ & 0xf) : LUA_TNONE);
}
