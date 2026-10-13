#include "lstate.h"
#include "lobject_tags.h"
#include "fn_luaS_new.h"
#include "fn_luaH_getshortstr.h"

const char *luaT_objtypename (lua_State *L, const TValue *o) {
  Table *mt;
  if ((o->tt_ == LUA_VTABLE && (mt = ((Table *)o->value_.gc)->metatable) != NULL) ||
      (o->tt_ == LUA_VUSERDATA && (mt = ((struct Udata *)o->value_.gc)->metatable) != NULL)) {
    const TValue *name = luaH_getshortstr(mt, luaS_new(L, "__name"));
    if ((name->tt_ & 0xf) == 4)  /* is '__name' a string? */
      return ((TString *)name->value_.gc)->contents;  /* use it as type name */
  }
  return luaT_typenames_[(o->tt_ & 0xf) + 1];  /* else use standard type name */
}
