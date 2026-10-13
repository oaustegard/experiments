#include "lstate.h"
#include "LClosure.h"
#include "cclosure.h"
#include "lua.h"

static UpVal **getupvalref (lua_State *L, int fidx, int n, LClosure **pf);

LUA_API void *lua_upvalueid (lua_State *L, int fidx, int n) {
  TValue *fi = index2value(L, fidx);
  switch (fi->tt_ & 0x3f) {
    case 6: {
      return *getupvalref(L, fidx, n, NULL);
    }
    case 0x26: {
      CClosure *f = (CClosure *)fi->value_.gc;
      if (1 <= n && n <= f->nupvalues)
        return &f->upvalue[n - 1];
    }
    /* FALLTHROUGH */
    case 0x16:
      return NULL;
    default: {
      return NULL;
    }
  }
}
