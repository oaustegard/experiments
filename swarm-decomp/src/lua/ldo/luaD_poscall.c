#include "lstate.h"
#include "fn_luaD_poscall.h"
#include "rethook.h"
#include "moveresults.h"

#define hastclose(n) ((n) < -1)

void luaD_poscall (lua_State *L, CallInfo *ci, int nres) {
  int wanted = ci->nresults;
  if (L->hookmask && __builtin_expect(!hastclose(wanted), 1))
    rethook(L, ci, nres);
  moveresults(L, ci->func, nres, wanted);
  L->ci = ci->previous;
}
