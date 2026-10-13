#include "lstate.h"
#include "fn_luaD_poscall.h"
#include "fn_luaD_hook.h"
#include "fn_luaD_growstack.h"
#include "fn_luaC_step.h"
#include "fn_luaE_extendCI.h"
#include "fn_luaD_pretailcall.h"
#include "tryfuncTM.h"

#define LUA_VCCL 0x26
#define LUA_VLCF_ 0x16
#define LUA_VLCL_ 0x06
#define CIST_C 2
#define CIST_TAIL 0x20
#define l_unlikely(x) __builtin_expect(((x) != 0), 0)
#define next_ci(L) ((L)->ci->next ? (L)->ci->next : luaE_extendCI(L))

int luaD_pretailcall (lua_State *L, CallInfo *ci, StkId func, int narg1, int delta) {
  lua_CFunction f;
retry:
  switch (func->tt_ & 0x3f) {
    case LUA_VLCF_:
      f = func->value_.f;
      goto Cfunc;
    case LUA_VCCL:
      f = ((CClosure *)func->value_.gc)->f;
      goto Cfunc;
    case LUA_VLCL_: {
      Proto *p = ((LClosure *)func->value_.gc)->p;
      int fsize = p->maxstacksize;
      int nfixparams = p->numparams;
      int i;
      if (l_unlikely(L->stack_last - L->top <= fsize - delta)) {
        ptrdiff_t t = (char *)func - (char *)L->stack;
        if (G(L)->GCdebt > 0)
          luaC_step(L);
        luaD_growstack(L, fsize - delta, 1);
        func = (StkId)((char *)L->stack + t);
      }
      ci->func -= delta;
      for (i = 0; i < narg1; i++) {
        (ci->func + i)->value_ = (func + i)->value_;
        (ci->func + i)->tt_ = (func + i)->tt_;
      }
      func = ci->func;
      for (; narg1 <= nfixparams; narg1++)
        (func + narg1)->tt_ = 0;
      ci->top = func + 1 + fsize;
      ci->u.l.savedpc = p->code;
      ci->callstatus |= CIST_TAIL;
      L->top = func + narg1;
      return -1;
    }
    default:
      narg1++;
      func = tryfuncTM(L, func);
      goto retry;
  }
Cfunc: {
    int n;
    CallInfo *nci;
    StkId ctop;
    if (l_unlikely(L->stack_last - L->top <= 20)) {
      ptrdiff_t t = (char *)func - (char *)L->stack;
      if (G(L)->GCdebt > 0)
        luaC_step(L);
      luaD_growstack(L, 20, 1);
      func = (StkId)((char *)L->stack + t);
    }
    ctop = L->top + 20;
    nci = next_ci(L);
    nci->nresults = -1;
    nci->callstatus = CIST_C;
    nci->func = func;
    nci->top = ctop;
    L->ci = nci;
    if (l_unlikely(L->hookmask & 1)) {
      int narg = (int)(L->top - func) - 1;
      luaD_hook(L, 0, -1, 1, narg);
    }
    n = (*f)(L);
    luaD_poscall(L, nci, n);
    return n;
  }
}
