#include "lstate.h"
#include "lua.h"

LUA_API void *lua_touserdata (lua_State *L, int idx) {
  const TValue *o = index2value(L, idx);
  void *p = o->value_.p;
  switch (o->tt_ & 0xf) {
    case 7: {
      Udata *u = (Udata *)p;
      return (char *)u + ((u->nuvalue == 0) ? 0x20 : 0x28 + sizeof(UValue) * u->nuvalue);
    }
    case 2: return p;
    default: return NULL;
  }
}
