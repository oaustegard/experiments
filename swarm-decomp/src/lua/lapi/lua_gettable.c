#include "lstate.h"
#include "fn_luaH_get.h"
#include "fn_luaV_finishget.h"
#include "lua.h"

static TValue *index2value (lua_State *L, int idx);

LUA_API int lua_gettable (lua_State *L, int idx) {
  const TValue *slot;
  TValue *t;
  t = index2value(L, idx);
  if (t->tt_ == 0x45 ? (slot = luaH_get((Table *)t->value_.gc, L->top - 1), (slot->tt_ & 0xf) != 0)
                     : (slot = NULL, 0)) {
    L->top[-1].value_ = slot->value_;
    L->top[-1].tt_ = slot->tt_;
  }
  else
    luaV_finishget(L, t, L->top - 1, L->top - 1, slot);
  return L->top[-1].tt_ & 0xf;
}
