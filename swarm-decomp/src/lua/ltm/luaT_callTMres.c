#include "lstate.h"
#include "lobject_tags.h"
#include "fn_luaD_call.h"

void luaT_callTMres (lua_State *L, const TValue *f, const TValue *p1,
                     const TValue *p2, StkId res) {
  ptrdiff_t result = res - L->stack;
  StkId func = L->top;
  func->value_ = f->value_; func->tt_ = f->tt_;
  (func+1)->value_ = p1->value_; (func+1)->tt_ = p1->tt_;
  (func+2)->value_ = p2->value_; (func+2)->tt_ = p2->tt_;
  L->top += 3;
  if (!(L->ci->callstatus & (2 | 8)))  /* Lua code? */
    luaD_call(L, func, 1);
  else
    luaD_callnoyield(L, func, 1);
  res = L->stack + result;
  L->top--;
  res->value_ = L->top->value_; res->tt_ = L->top->tt_;
}
