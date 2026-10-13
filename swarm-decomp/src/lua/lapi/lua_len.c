#include "lstate.h"
#include "fn_luaV_objlen.h"
#include "lua.h"

LUA_API void lua_len (lua_State *L, int idx) {
  TValue *t;
  t = index2value(L, idx);
  luaV_objlen(L, L->top, t);
  L->top++;
}
