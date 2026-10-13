#include <setjmp.h>
#include <stdlib.h>
#include "lstate.h"
#include "lua_longjmp.h"
#include "lobject_setters.h"
#include "fn_luaE_resetthread.h"
#include "fn_luaD_throw.h"

_Noreturn void luaD_throw (lua_State *L, int errcode) {
  if (L->errorJmp) {
    struct lua_longjmp *lj = (struct lua_longjmp *)L->errorJmp;
    lj->status = errcode;
    _longjmp(lj->b, 1);
  }
  else {
    global_State *g = G(L);
    errcode = luaE_resetthread(L, errcode);
    if (g->mainthread->errorJmp != NULL) {
      setobjs2s(L, g->mainthread->top++, L->top - 1);
      luaD_throw(g->mainthread, errcode);
    }
    else {
      if (g->panic) {
        g->panic(L);
      }
      abort();
    }
  }
}
