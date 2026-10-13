#include "lstate.h"
#include "lobject_setters.h"
#include "fn_luaF_closeupval.h"
#include "fn_luaT_gettmbyobj.h"
#include "fn_luaD_seterrorobj.h"
#include "fn_luaD_call.h"
#include "fn_luaD_callnoyield.h"
#include "fn_luaF_close.h"

#define CLOSEKTOP (-1)
#define MAXDELTA ((1 << 16) - 1)
#define TBCDELTA(o) (*(unsigned short *)((char *)(o) + 10))

StkId luaF_close (lua_State *L, StkId level, int status, int yy) {
  ptrdiff_t levelrel = (char *)level - (char *)L->stack;
  luaF_closeupval(L, level);
  while (L->tbclist >= level) {
    StkId tbc = L->tbclist;
    StkId p = tbc - TBCDELTA(tbc);
    while (p > L->stack && TBCDELTA(p) == 0)
      p -= MAXDELTA;
    L->tbclist = p;
    {
      TValue *uv = s2v(tbc);
      TValue *errobj;
      if (status == CLOSEKTOP)
        errobj = &L->l_G->nilvalue;
      else {
        errobj = s2v(tbc + 1);
        luaD_seterrorobj(L, status, tbc + 1);
      }
      {
        StkId top = L->top;
        const TValue *tm = luaT_gettmbyobj(L, uv, TM_CLOSE);
        setobjs2s(L, top, tm);
        setobjs2s(L, top + 1, uv);
        setobjs2s(L, top + 2, errobj);
        L->top = top + 3;
        if (yy)
          luaD_call(L, top, 0);
        else
          luaD_callnoyield(L, top, 0);
      }
    }
    level = (StkId)((char *)L->stack + levelrel);
  }
  return level;
}
