#include "lstate.h"
#include "LClosure.h"
#include "fn_luaD_hook.h"
#include "fn_luaD_throw.h"
#include "fn_luaG_getfuncline.h"
#include "fn_luaG_traceexec.h"
#include "changedline.h"
#include "ldebug_ops.h"

#define LUA_MASKCALL	(1 << 0)
#define LUA_MASKRET	(1 << 1)
#define LUA_MASKLINE	(1 << 2)
#define LUA_MASKCOUNT	(1 << 3)
#define LUA_HOOKLINE	2
#define LUA_HOOKCOUNT	3
#define LUA_YIELD	1
#define CIST_HOOKYIELD	(1 << 6)
#define ci_func(ci)	((LClosure *)((ci)->func->value_.gc))
#define testITMode(m)	(luaP_opmodes[m] & (1 << 5))
#define isIT(i)		(testITMode(DGET_OPCODE(i)) && DGETARG_B(i) == 0)
#define resethookcount(L)	(L->hookcount = L->basehookcount)
#define pcRel(pc, p)	((int)((pc) - (p)->code) - 1)

int luaG_traceexec (lua_State *L, const Instruction *pc) {
  CallInfo *ci = L->ci;
  lu_byte mask = L->hookmask;
  const Proto *p = ci_func(ci)->p;
  int counthook;
  if (!(mask & (LUA_MASKLINE | LUA_MASKCOUNT))) {  /* no hooks? */
    ci->u.l.trap = 0;  /* don't need to stop again */
    return 0;  /* turn off 'trap' */
  }
  pc++;  /* reference is always next instruction */
  ci->u.l.savedpc = pc;  /* save 'pc' */
  counthook = ((mask & LUA_MASKCOUNT) && --L->hookcount == 0);
  if (counthook)
    resethookcount(L);  /* reset count */
  else if (!(mask & LUA_MASKLINE))
    return 1;  /* no line hook and count != 0; nothing to be done now */
  if (ci->callstatus & CIST_HOOKYIELD) {  /* called hook last time? */
    ci->callstatus &= ~CIST_HOOKYIELD;  /* erase mark */
    return 1;  /* do not call hook again (VM yielded, so it did not move) */
  }
  if (!isIT(*(ci->u.l.savedpc - 1)))  /* top not valid? */
    L->top = ci->top;  /* correct top */
  if (counthook)
    luaD_hook(L, LUA_HOOKCOUNT, -1, 0, 0);  /* call count hook */
  if (mask & LUA_MASKLINE) {
    /* 'L->oldpc' may be invalid; use zero in this case */
    int oldpc = (L->oldpc < p->sizecode) ? L->oldpc : 0;
    int npci = pcRel(pc, p);
    if (npci <= oldpc ||  /* call hook before the instruction? (jump back) */
        changedline(p, oldpc, npci)) {  /* or when enter new line */
      int newline = luaG_getfuncline(p, npci);
      luaD_hook(L, LUA_HOOKLINE, newline, 0, 0);  /* call line hook */
    }
    L->oldpc = npci;  /* 'pc' of last call to line hook */
  }
  if (L->status == LUA_YIELD) {  /* did hook yield? */
    if (counthook)
      L->hookcount = 1;  /* undo decrement to zero */
    ci->callstatus |= CIST_HOOKYIELD;  /* mark that it yielded */
    luaD_throw(L, LUA_YIELD);
  }
  return 1;  /* keep 'trap' on */
}
