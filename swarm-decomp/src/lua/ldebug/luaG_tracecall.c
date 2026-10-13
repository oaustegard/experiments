#include "lstate.h"
#include "LClosure.h"
#include "fn_luaD_hookcall.h"

#define CIST_HOOKYIELD	(1<<6)

int luaG_tracecall (lua_State *L) {
  CallInfo *ci = L->ci;
  Proto *p = ((LClosure *)(ci->func->value_.gc))->p;
  ci->u.l.trap = 1;
  if (ci->u.l.savedpc == p->code) {  /* first instruction (not resuming)? */
    if (p->is_vararg)
      return 0;  /* hooks will start at VARARGPREP instruction */
    else if (!(ci->callstatus & CIST_HOOKYIELD))  /* not yieded? */
      luaD_hookcall(L, ci);  /* check 'call' hook */
  }
  return 1;  /* keep 'trap' on */
}
