#include "lstate.h"
#include "Proto.h"
#include "lobject_tags.h"
#include "lobject_setters.h"
#include "fn_luaD_growstack.h"

void luaT_adjustvarargs (lua_State *L, int nfixparams, CallInfo *ci,
                         const Proto *p) {
  int i;
  int actual = (int)(L->top - ci->func) - 1;  /* number of arguments */
  int nextra = actual - nfixparams;  /* number of extra arguments */
  ci->u.l.nextraargs = nextra;
  if (L->stack_last - L->top <= p->maxstacksize + 1)
    luaD_growstack(L, p->maxstacksize + 1, 1);
  /* copy function to the top of the stack */
  setobjs2s(L, L->top++, ci->func);
  /* move fixed parameters to the top of the stack */
  for (i = 1; i <= nfixparams; i++) {
    setobjs2s(L, L->top++, ci->func + i);
    setnilvalue(s2v(ci->func + i));  /* erase original parameter (for GC) */
  }
  ci->func += actual + 1;
  ci->top += actual + 1;
}
