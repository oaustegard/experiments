#include "lstate.h"
#include "fn_luaM_realloc_.h"
#include "fn_luaD_throw.h"
#include "correctstack.h"
#include "fn_luaD_reallocstack.h"

#define EXTRA_STACK 5

int luaD_reallocstack (lua_State *L, int newsize, int raiseerror) {
  int oldsize = (int)(L->stack_last - L->stack);
  int i;
  StkId newstack;
  lu_byte oldgcstop = G(L)->gcstopem;
  L->top = (StkId)((char *)L->top - (char *)L->stack);
  L->tbclist = (StkId)((char *)L->tbclist - (char *)L->stack);
  {
    UpVal *up;
    CallInfo *ci;
    for (up = L->openupval; up != NULL; up = up->u.open.next)
      up->v.p = (TValue *)((char *)up->v.p - (char *)L->stack);
    for (ci = L->ci; ci != NULL; ci = ci->previous) {
      ci->top = (StkId)((char *)ci->top - (char *)L->stack);
      ci->func = (StkId)((char *)ci->func - (char *)L->stack);
    }
  }
  G(L)->gcstopem = 1;
  newstack = (StkId)luaM_realloc_(L, L->stack, (size_t)(oldsize + EXTRA_STACK) * sizeof(TValue), (size_t)(newsize + EXTRA_STACK) * sizeof(TValue));
  G(L)->gcstopem = oldgcstop;
  if (__builtin_expect(newstack == NULL, 0)) {
    correctstack(L);
    if (raiseerror)
      luaD_throw(L, 4);
    return 0;
  } else {
    L->stack = newstack;
    correctstack(L);
    L->stack_last = L->stack + newsize;
    for (i = oldsize + EXTRA_STACK; i < newsize + EXTRA_STACK; i++)
      setnilvalue(newstack + i);
  }
  return 1;
}
