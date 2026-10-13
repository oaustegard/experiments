#include "lstate.h"
#include "Udata.h"
#include "fn_luaC_barrierback_.h"
#include "lua.h"

static TValue *index2value (lua_State *L, int idx);

#define uvalue(o) ((Udata *)(o)->value_.gc)

LUA_API int lua_setiuservalue (lua_State *L, int idx, int n) {
  TValue *o;
  int res;
  o = index2value(L, idx);
  if ((unsigned int)(n - 1) >= (unsigned int)uvalue(o)->nuvalue)
    res = 0;
  else {
    TValue *io1 = &uvalue(o)->uv[n - 1].uv;
    const TValue *io2 = L->top - 1;
    io1->value_ = io2->value_;
    io1->tt_ = io2->tt_;
    if (io2->tt_ & (1 << 6)) {
      if (((GCObject *)o->value_.gc)->marked & (1 << 5)) {
        if (((GCObject *)io2->value_.gc)->marked & (3 << 3))
          luaC_barrierback_(L, (GCObject *)o->value_.gc);
      }
    }
    res = 1;
  }
  L->top--;
  return res;
}
