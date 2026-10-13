#include "lstate.h"
#include "fn_luaC_barrierback_.h"
#include "fn_luaH_getstr.h"
#include "fn_luaS_new.h"
#include "fn_luaV_finishset.h"

static void auxsetstr (lua_State *L, const TValue *t, const char *k) {
  const TValue *slot = NULL;
  TString *str = luaS_new(L, k);
  if (t->tt_ == 0x45 &&
      (slot = luaH_getstr((Table *)t->value_.gc, str), (slot->tt_ & 0xf) != 0)) {
    TValue *v = L->top - 1;
    TValue *s = (TValue *)slot;
    s->value_ = v->value_;
    s->tt_ = v->tt_;
    if ((v->tt_ & 0x40) && (t->value_.gc->marked & 0x20) && (v->value_.gc->marked & 0x18))
      luaC_barrierback_(L, t->value_.gc);
    L->top--;
  }
  else {
    L->top->value_.gc = (GCObject *)str;
    L->top->tt_ = str->tt | 0x40;
    L->top++;
    luaV_finishset(L, t, L->top - 1, L->top - 2, slot);
    L->top -= 2;
  }
}
