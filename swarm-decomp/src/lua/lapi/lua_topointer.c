#include "lstate.h"
#include "lua.h"

LUA_API const void *lua_topointer (lua_State *L, int idx) {
  const TValue *o = index2value(L, idx);
  lu_byte tt = o->tt_;
  switch (tt & 0x3f) {
    case 0x16: return (const void *)(size_t)o->value_.f;
    case 7:
    case 2: {
      Udata *u = (Udata *)o->value_.gc;
      if ((tt & 0xf) == 2) return o->value_.p;
      if ((tt & 0xf) == 7)
        return (char *)u + (u->nuvalue == 0 ? 0x20 : 0x28 + sizeof(UValue) * u->nuvalue);
      else
        return NULL;
    }
    default: {
      if (tt & 0x40)
        return o->value_.gc;
      else
        return NULL;
    }
  }
}
