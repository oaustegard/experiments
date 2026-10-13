#include "lstate.h"
#include "fn_luaH_getn.h"
#include "lua.h"

LUA_API lua_Unsigned lua_rawlen (lua_State *L, int idx) {
  const TValue *o = index2value(L, idx);
  switch (o->tt_ & 0x3f) {
    case 4: return ((TString *)o->value_.gc)->shrlen;
    case 0x14: return ((TString *)o->value_.gc)->u.lnglen;
    case 7: return ((Udata *)o->value_.gc)->len;
    case 5: return luaH_getn((Table *)o->value_.gc);
    default: return 0;
  }
}
