#include "lstate.h"
#include "fn_luaH_getstr.h"
#include "fn_luaS_new.h"
#include "fn_luaV_finishget.h"
#include "lua.h"

LUA_API int lua_getfield (lua_State *L, int idx, const char *k) {
  const TValue *t = index2value(L, idx);
  const TValue *slot = NULL;
  TString *str = luaS_new(L, k);
  if (t->tt_ == 0x45 &&
      (slot = luaH_getstr((Table *)t->value_.gc, str), (slot->tt_ & 0xf) != 0)) {
    L->top->value_ = slot->value_;
    L->top->tt_ = slot->tt_;
    L->top++;
  }
  else {
    L->top->value_.gc = (GCObject *)str;
    L->top->tt_ = str->tt | 0x40;
    L->top++;
    luaV_finishget(L, t, L->top - 1, L->top - 1, slot);
  }
  return L->top[-1].tt_ & 0xf;
}
