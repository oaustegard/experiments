#include "lstate.h"
#include "StackValue.h"
#include "fn_luaO_arith.h"
#include "lua.h"

LUA_API void lua_arith (lua_State *L, int op) {
  StackValue *top = (StackValue *)L->top;
  if (op == LUA_OPUNM || op == LUA_OPBNOT) {
    TValue *io1 = &top->val;
    const TValue *io2 = &(top - 1)->val;
    io1->value_ = io2->value_;
    io1->tt_ = io2->tt_;
    top = top + 1;
    L->top = (StkId)top;
  }
  luaO_arith(L, op, &(top - 2)->val, &(top - 1)->val, &(top - 2)->val);
  L->top--;
}
