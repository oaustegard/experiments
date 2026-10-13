#include "lstate.h"
#include "fn_luaF_close.h"
#include "rethook.h"
#include "moveresults.h"

#define CIST_CLSRET 0x200
#define copyobj(d, s) ((d)->value_ = (s)->value_, (d)->tt_ = (s)->tt_)

static void moveresults (lua_State *L, StkId res, int nres, int wanted) {
  StkId firstresult;
  int i;
  switch (wanted) {
    case 0:
      L->top = res;
      return;
    case 1:
      if (nres == 0)
        setnilvalue(res);
      else
        copyobj(res, L->top - nres);
      L->top = res + 1;
      return;
    case -1:
      wanted = nres;
      break;
    default:
      if (wanted < -1) {
        L->ci->callstatus |= CIST_CLSRET;
        L->ci->u2.funcidx = nres;
        res = luaF_close(L, res, -1, 1);
        L->ci->callstatus &= ~CIST_CLSRET;
        if (L->hookmask) {
          ptrdiff_t savedres = (char *)res - (char *)L->stack;
          rethook(L, L->ci, nres);
          res = (StkId)((char *)L->stack + savedres);
        }
        wanted = -3 - wanted;
        if (wanted == -1)
          wanted = nres;
      }
      break;
  }
  firstresult = L->top - nres;
  if (nres > wanted)
    nres = wanted;
  for (i = 0; i < nres; i++)
    copyobj(res + i, firstresult + i);
  for (; i < wanted; i++)
    setnilvalue(res + i);
  L->top = res + wanted;
}
