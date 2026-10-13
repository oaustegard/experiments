#include <setjmp.h>
#include "lstate.h"
#include "lua_longjmp.h"
#include "Pfunc.h"

int luaD_rawrunprotected (lua_State *L, Pfunc f, void *ud) {
  unsigned int oldnCcalls = L->nCcalls;
  struct lua_longjmp lj;
  lj.status = 0;
  lj.previous = (struct lua_longjmp *)L->errorJmp;
  L->errorJmp = &lj;
  if (_setjmp(lj.b) == 0) {
    (*f)(L, ud);
  }
  L->errorJmp = lj.previous;
  L->nCcalls = oldnCcalls;
  return lj.status;
}
