#include "lstate.h"
#include "CallS.h"
#include "fn_luaD_callnoyield.h"

static void f_call (lua_State *L, void *ud) {
  struct CallS *c = (struct CallS *)ud;
  luaD_callnoyield(L, c->func, c->nresults);
}
