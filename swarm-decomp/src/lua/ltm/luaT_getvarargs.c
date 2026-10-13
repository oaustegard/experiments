#include "lstate.h"
#include "lobject_tags.h"
#include "fn_luaD_growstack.h"
#include "fn_luaC_step.h"

void luaT_getvarargs (lua_State *L, CallInfo *ci, StkId where, int wanted) {
  int i;
  int nextra = ci->u.l.nextraargs;
  if (wanted < 0) {
    wanted = nextra;  /* get all extra arguments available */
    if (__builtin_expect(L->stack_last - L->top <= nextra, 0)) {
      ptrdiff_t t = where - L->stack;
      if (G(L)->GCdebt > 0)
        luaC_step(L);
      luaD_growstack(L, nextra, 1);
      where = L->stack + t;
    }
    L->top = where + nextra;  /* next instruction will need top */
  }
  for (i = 0; i < wanted && i < nextra; i++) {
    StkId src = ci->func - nextra + i;
    where[i].value_ = src->value_; where[i].tt_ = src->tt_;
  }
  for (; i < wanted; i++)   /* complete required results with nil */
    where[i].tt_ = 0;
}
