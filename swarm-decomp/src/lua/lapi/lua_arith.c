#include "lstate.h"
#include "fn_luaO_arith.h"
#include "lua.h"

LUA_API void lua_arith (lua_State *L, int op) {
  StkId top = L->top;
  if (op == LUA_OPUNM || op == LUA_OPBNOT) {
    top[0] = top[-1];
    L->top = top = top + 1;
  }
  luaO_arith(L, op, top - 2, top - 1, top - 2);
  L->top--;
}
