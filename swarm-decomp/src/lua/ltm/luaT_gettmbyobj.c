#include "lstate.h"
#include "fn_luaH_getshortstr.h"

const TValue *luaT_gettmbyobj (lua_State *L, const TValue *o, TMS event) {
  Table *mt;
  switch (o->tt_ & 0xf) {
    case LUA_TTABLE:
      mt = ((Table *)o->value_.gc)->metatable;
      break;
    case LUA_TUSERDATA:
      mt = ((struct Udata *)o->value_.gc)->metatable;
      break;
    default:
      mt = G(L)->mt[o->tt_ & 0xf];
  }
  return (mt ? luaH_getshortstr(mt, G(L)->tmname[event]) : &G(L)->nilvalue);
}
