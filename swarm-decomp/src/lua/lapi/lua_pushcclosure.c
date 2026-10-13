#include "lstate.h"
#include "fn_luaF_newCclosure.h"
#include "fn_luaC_step.h"
#include "lua.h"

LUA_API void lua_pushcclosure (lua_State *L, lua_CFunction fn, int n) {
  if (n == 0) {
    L->top->value_.f = fn;
    L->top->tt_ = 0x16;
    L->top++;
  }
  else {
    CClosure *cl;
    cl = luaF_newCclosure(L, n);
    cl->f = fn;
    L->top -= n;
    while (n--) {
      TValue *io1 = &cl->upvalue[n];
      const TValue *io2 = L->top + n;
      io1->value_ = io2->value_;
      io1->tt_ = io2->tt_;
    }
    L->top->value_.gc = (GCObject *)cl;
    L->top->tt_ = 0x66;
    L->top++;
    if (L->l_G->GCdebt > 0)
      luaC_step(L);
  }
}
