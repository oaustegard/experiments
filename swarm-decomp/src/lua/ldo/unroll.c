#include "lstate.h"
#include "unroll.h"
#include "fn_luaV_finishOp.h"
#include "fn_luaV_execute.h"
#include "finishCcall.h"

#define isLua(ci) (!((ci)->callstatus & 2))

static void unroll (lua_State *L, void *ud) {
  CallInfo *ci;
  while ((ci = L->ci) != &L->base_ci) {
    if (!isLua(ci))
      finishCcall(L, ci);
    else {
      luaV_finishOp(L);
      luaV_execute(L, ci);
    }
  }
}
