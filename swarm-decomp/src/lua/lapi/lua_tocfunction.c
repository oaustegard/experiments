#include "lstate.h"
#include "lua.h"

LUA_API lua_CFunction lua_tocfunction (lua_State *L, int idx) {
  const TValue *o = index2value(L, idx);
  if (o->tt_ == 0x16)
    return o->value_.f;
  else if (o->tt_ == 0x66)
    return ((CClosure *)o->value_.gc)->f;
  else
    return NULL;
}
