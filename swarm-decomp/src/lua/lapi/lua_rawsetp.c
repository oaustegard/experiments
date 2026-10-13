#include "lstate.h"
#include "fn_luaH_set.h"
#include "fn_luaC_barrierback_.h"
#include "lua.h"

static TValue *index2value (lua_State *L, int idx);

LUA_API void lua_rawsetp (lua_State *L, int idx, const void *p) {
  Table *t;
  TValue k;
  k.value_.p = (void *)p;
  k.tt_ = 2;
  t = (Table *)index2value(L, idx)->value_.gc;
  luaH_set(L, t, &k, L->top - 1);
  t->flags &= ~0x3f;
  if ((L->top - 1)->tt_ & 0x40) {
    if ((t->marked & 0x20) && ((L->top - 1)->value_.gc->marked & 0x18))
      luaC_barrierback_(L, (GCObject *)t);
  }
  L->top -= 1;
}
