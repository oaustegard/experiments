#include "lstate.h"
#include "fn_luaD_precall.h"
#include "fn_luaD_poscall.h"
#include "fn_luaD_hook.h"
#include "fn_luaD_growstack.h"
#include "fn_luaC_step.h"
#include "fn_luaE_extendCI.h"
#include "tryfuncTM.h"

#define LUA_VCCL 0x26
#define LUA_VLCF_ 0x16
#define LUA_VLCL_ 0x06
#define l_unlikely(x) __builtin_expect(((x) != 0), 0)
#define next_ci(L) ((L)->ci->next ? (L)->ci->next : luaE_extendCI(L))

CallInfo *luaD_precall (lua_State *L, StkId func, int nresults) {
  CallInfo *ci;
retry:
  switch (func->tt_ & 0x3f) {
    case LUA_VLCF_: {
      lua_CFunction f = func->value_.f;
      int n;
      StkId ctop;
      if (l_unlikely(L->stack_last - L->top <= 20)) {
        ptrdiff_t t = (char *)func - (char *)L->stack;
        if (G(L)->GCdebt > 0)
          luaC_step(L);
        luaD_growstack(L, 20, 1);
        func = (StkId)((char *)L->stack + t);
      }
      ctop = L->top + 20;
      ci = next_ci(L);
      ci->func = func;
      ci->nresults = nresults;
      ci->callstatus = 2;
      ci->top = ctop;
      L->ci = ci;
      if (l_unlikely(L->hookmask & 1)) {
        int narg = (int)(L->top - func) - 1;
        luaD_hook(L, 0, -1, 1, narg);
      }
      n = (*f)(L);
      luaD_poscall(L, ci, n);
      return NULL;
    }
    case LUA_VCCL: {
      lua_CFunction f = ((CClosure *)func->value_.gc)->f;
      int n;
      StkId ctop;
      if (l_unlikely(L->stack_last - L->top <= 20)) {
        ptrdiff_t t = (char *)func - (char *)L->stack;
        if (G(L)->GCdebt > 0)
          luaC_step(L);
        luaD_growstack(L, 20, 1);
        func = (StkId)((char *)L->stack + t);
      }
      ctop = L->top + 20;
      ci = next_ci(L);
      ci->func = func;
      ci->nresults = nresults;
      ci->callstatus = 2;
      ci->top = ctop;
      L->ci = ci;
      if (l_unlikely(L->hookmask & 1)) {
        int narg = (int)(L->top - func) - 1;
        luaD_hook(L, 0, -1, 1, narg);
      }
      n = (*f)(L);
      luaD_poscall(L, ci, n);
      return NULL;
    }
    case LUA_VLCL_: {
      Proto *p = ((LClosure *)func->value_.gc)->p;
      int narg = (int)(L->top - func) - 1;
      int fsize = p->maxstacksize;
      int nfixparams = p->numparams;
      if (l_unlikely(L->stack_last - L->top <= fsize)) {
        ptrdiff_t t = (char *)func - (char *)L->stack;
        if (G(L)->GCdebt > 0)
          luaC_step(L);
        luaD_growstack(L, fsize, 1);
        func = (StkId)((char *)L->stack + t);
      }
      StkId ltop = func + 1 + fsize;
      ci = next_ci(L);
      ci->func = func;
      ci->callstatus = 0;
      ci->nresults = nresults;
      ci->top = ltop;
      L->ci = ci;
      ci->u.l.savedpc = p->code;
      for (; narg < nfixparams; narg++)
        setnilvalue(s2v(L->top++));
      return ci;
    }
    default:
      func = tryfuncTM(L, func);
      goto retry;
  }
}
